from pathlib import Path
import warnings, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT=Path(__file__).parent; PLOTS=ROOT/'plots'; PLOTS.mkdir(exist_ok=True)

def main():
    # Required single raw load
    df=sns.load_dataset('titanic')
    df.to_csv(ROOT/'titanic.csv',index=False)
    with open(ROOT/'profile.txt','w') as f:
        f.write('INFO\n'); df.info(buf=f); f.write('\nDESCRIBE\n'+df.describe(include='all').to_string()+'\nSHAPE\n'+str(df.shape))
    miss=(df.isna().mean()*100).loc[lambda x:x>0]
    (ROOT/'missing_values.csv').write_text(miss.to_csv(header=['missing_pct']))
    # EDA cleaning per assignment threshold
    cleaned=df.copy()
    if miss.get('embarked',0)<5: cleaned=cleaned.dropna(subset=['embarked'])
    if 5<=miss.get('age',0)<=30: cleaned['age']=cleaned['age'].fillna(cleaned['age'].median())
    if miss.get('deck',0)>30: cleaned=cleaned.drop(columns=['deck'])
    cleaned=cleaned.dropna(subset=['embarked'])
    cleaned.to_csv(ROOT/'titanic_cleaned.csv',index=False)
    # Univariate
    stats={}
    for col in ['age','fare']:
        q1,q3=cleaned[col].quantile([.25,.75]); iqr=q3-q1
        stats[col]={'q1':q1,'q3':q3,'iqr':iqr,'outliers':int(((cleaned[col]<q1-1.5*iqr)|(cleaned[col]>q3+1.5*iqr)).sum())}
        fig,ax=plt.subplots(); sns.histplot(cleaned[col],kde=True,ax=ax); ax.set_title(f'{col.title()} Histogram'); fig.savefig(PLOTS/f'{col}_hist.png',bbox_inches='tight'); plt.close(fig)
        fig,ax=plt.subplots(); sns.boxplot(x=cleaned[col],ax=ax); ax.set_title(f'{col.title()} Box Plot'); fig.savefig(PLOTS/f'{col}_box.png',bbox_inches='tight'); plt.close(fig)
    fare_mean=cleaned.fare.mean(); fare_med=cleaned.fare.median(); fare_mode=cleaned.fare.mode().iloc[0]
    skew='right-skewed' if fare_mean>fare_med else ('left-skewed' if fare_mean<fare_med else 'approximately symmetric')
    # Bivariate
    surv_sex=cleaned.groupby('sex').survived.mean().reset_index(name='survival_rate')
    surv_pclass=cleaned.groupby('pclass').survived.mean().reset_index(name='survival_rate')
    surv_both=cleaned.groupby(['sex','pclass']).survived.mean().reset_index(name='survival_rate')
    surv_sex.to_csv(ROOT/'survival_by_sex.csv',index=False); surv_pclass.to_csv(ROOT/'survival_by_pclass.csv',index=False); surv_both.to_csv(ROOT/'survival_by_sex_pclass.csv',index=False)
    corr_cols=['survived','pclass','age','sibsp','parch','fare']; corr=cleaned[corr_cols].corr(); corr.to_csv(ROOT/'correlation.csv')
    fig,ax=plt.subplots(figsize=(7,5)); sns.heatmap(corr,annot=True,cmap='vlag',center=0,ax=ax); ax.set_title('6x6 Correlation Heatmap'); fig.savefig(PLOTS/'correlation_heatmap.png',bbox_inches='tight'); plt.close(fig)
    pairs=[]
    for i,a in enumerate(corr_cols):
        for b in corr_cols[i+1:]: pairs.append((abs(corr.loc[a,b]),a,b,corr.loc[a,b]))
    top2=sorted(pairs,reverse=True)[:2]
    # 4+ data-story charts
    fig,ax=plt.subplots(); sns.barplot(data=surv_sex,x='sex',y='survival_rate',ax=ax); ax.set_title('Survival Rate by Sex'); fig.savefig(PLOTS/'story1_sex.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(); sns.barplot(data=surv_pclass,x='pclass',y='survival_rate',ax=ax); ax.set_title('Survival Rate by Passenger Class'); fig.savefig(PLOTS/'story2_class.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(); sns.boxplot(data=cleaned,x='survived',y='fare',ax=ax); ax.set_title('Fare Distribution by Survival'); fig.savefig(PLOTS/'story3_fare_survival.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(); sns.boxplot(data=cleaned,x='survived',y='age',ax=ax); ax.set_title('Age Distribution by Survival'); fig.savefig(PLOTS/'story4_age_survival.png',bbox_inches='tight'); plt.close(fig)
    # exploratory z-score
    before=cleaned[['age','fare']].agg(['mean','std']); z=cleaned[['age','fare']].apply(lambda x:(x-x.mean())/x.std()); after=z.agg(['mean','std']); pd.concat({'before':before,'after':after},axis=1).to_csv(ROOT/'standardization_check.csv')
    # Model data: split first, preprocessing fit only train
    features=['pclass','age','sibsp','parch','fare','sex','embarked']; X=cleaned[features]; y=cleaned['survived']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    num=['pclass','age','sibsp','parch','fare']; cat=['sex','embarked']
    pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    models={'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),'Decision Tree':DecisionTreeClassifier(random_state=42,max_depth=5),'Random Forest':RandomForestClassifier(random_state=42,n_estimators=200)}
    results=[]; roc_data={}
    for name,est in models.items():
        pipe=Pipeline([('pre',pre),('model',est)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
        results.append([name,accuracy_score(yte,pred),precision_score(yte,pred),recall_score(yte,pred),f1_score(yte,pred),roc_auc_score(yte,prob)])
        roc_data[name]=(yte,prob)
        cm=confusion_matrix(yte,pred); pd.DataFrame(cm,index=['Actual 0','Actual 1'],columns=['Pred 0','Pred 1']).to_csv(ROOT/f'{name.lower().replace(" ","_")}_confusion_matrix.csv')
    metrics=pd.DataFrame(results,columns=['model','accuracy','precision','recall','f1','auc']); metrics.to_csv(ROOT/'classification_metrics.csv',index=False)
    # ROC combined
    fig,ax=plt.subplots()
    for name,(yy,prob) in roc_data.items():
        fpr,tpr,_=roc_curve(yy,prob); ax.plot(fpr,tpr,label=f'{name} AUC={roc_auc_score(yy,prob):.3f}')
    ax.plot([0,1],[0,1],linestyle='--'); ax.legend(); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_title('ROC Curves'); fig.savefig(PLOTS/'roc_curves.png',bbox_inches='tight'); plt.close(fig)
    # Decision tree visualization
    dt_pipe=Pipeline([('pre',pre),('model',DecisionTreeClassifier(random_state=42,max_depth=5))]); dt_pipe.fit(Xtr,ytr)
    feat=dt_pipe.named_steps['pre'].get_feature_names_out()
    fig,ax=plt.subplots(figsize=(20,10)); plot_tree(dt_pipe.named_steps['model'],feature_names=feat,class_names=['Not survived','Survived'],filled=True,max_depth=4,ax=ax); fig.savefig(PLOTS/'decision_tree.png',bbox_inches='tight'); plt.close(fig)
    # imbalance comparison using logistic regression
    base=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1000,random_state=42))]); base.fit(Xtr,ytr)
    balanced=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))]); balanced.fit(Xtr,ytr)
    smote=ImbPipeline([('pre',pre),('smote',SMOTE(random_state=42)),('model',LogisticRegression(max_iter=1000,random_state=42))]); smote.fit(Xtr,ytr)
    imrows=[]
    for n,p in [('baseline',base),('class_weight_balanced',balanced),('smote',smote)]:
        pr=p.predict(Xte); imrows.append([n,precision_score(yte,pr),recall_score(yte,pr),f1_score(yte,pr)])
    pd.DataFrame(imrows,columns=['strategy','precision','recall','f1']).to_csv(ROOT/'imbalance_comparison.csv',index=False)
    # RF GridSearch + OOB
    rf=RandomForestClassifier(random_state=42,oob_score=True,bootstrap=True)
    grid=GridSearchCV(rf,{'n_estimators':[100,200],'max_depth':[None,5,10],'max_features':['sqrt','log2']},cv=5,scoring='f1',n_jobs=-1)
    rf_pipe=Pipeline([('pre',pre),('model',grid)]); rf_pipe.fit(Xtr,ytr)
    best=rf_pipe.named_steps['model'].best_estimator_; grid_info={'best_params':rf_pipe.named_steps['model'].best_params_,'oob_score':best.oob_score_}; (ROOT/'rf_gridsearch.json').write_text(json.dumps(grid_info,indent=2))
    # regression fare from other features
    reg_features=['pclass','age','sibsp','parch','survived','sex','embarked']; Xr=cleaned[reg_features]; yr=cleaned['fare']; Xrt,Xrv,yrt,yrv=train_test_split(Xr,yr,test_size=.2,random_state=42)
    rn=['pclass','age','sibsp','parch','survived']; rc=['sex','embarked']; rpre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),rn),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),rc)])
    reg=Pipeline([('pre',rpre),('model',LinearRegression())]); reg.fit(Xrt,yrt); rp=reg.predict(Xrv); mae=mean_absolute_error(yrv,rp); rmse=np.sqrt(mean_squared_error(yrv,rp)); r2=r2_score(yrv,rp); n=len(yrv); p=reg.named_steps['pre'].transform(Xrv).shape[1]; adj=1-(1-r2)*(n-1)/(n-p-1)
    pd.DataFrame([{'MAE':mae,'RMSE':rmse,'R2':r2,'Adjusted_R2':adj}]).to_csv(ROOT/'regression_metrics.csv',index=False)
    residuals=yrv-rp; fig,ax=plt.subplots(); ax.scatter(rp,residuals); ax.axhline(0,linestyle='--'); ax.set_xlabel('Predicted Fare'); ax.set_ylabel('Residual'); ax.set_title('Fare Regression Residuals'); fig.savefig(PLOTS/'regression_residuals.png',bbox_inches='tight'); plt.close(fig)
    # Save best classifier pipeline based on F1
    best_name=metrics.sort_values('f1',ascending=False).iloc[0]['model']; best_pipe=Pipeline([('pre',pre),('model',models[best_name])]); best_pipe.fit(Xtr,ytr); dump(best_pipe,ROOT/'best_classifier_pipeline.joblib')
    reload_pipe=load(ROOT/'best_classifier_pipeline.joblib'); sample=reload_pipe.predict(Xte.iloc[:3]); (ROOT/'reload_check.txt').write_text('Reloaded pipeline predictions on raw rows: '+str(sample.tolist()))
    # auto report
    story=[]
    story.append(f'Fare mean={fare_mean:.3f}, median={fare_med:.3f}, mode={fare_mode:.3f}; distribution conclusion: {skew}.')
    for c in ['age','fare']: story.append(f"{c}: IQR-based outlier count = {stats[c]['outliers']}.")
    story.append('Top two absolute off-diagonal correlations: '+', '.join([f'{a} vs {b}: {v:.3f}' for _,a,b,v in top2])+'.')
    story.append('Multivariate chart 1: survival differs by sex, providing evidence of a strong sex-associated survival pattern.')
    story.append('Multivariate chart 2: survival varies across passenger class, showing socioeconomic/class association.')
    story.append('Multivariate chart 3: fare distributions differ between survivors and non-survivors, linking ticket price/class to outcomes.')
    story.append('Multivariate chart 4: age distributions differ across survival groups, showing an age-associated pattern.')
    story.append(f'Stratification preserves the observed target-class proportions in train and test: train={ytr.mean():.3f}, test={yte.mean():.3f}.')
    story.append(f'Classifier selected by highest test F1: {best_name}. Metrics are recorded in classification_metrics.csv.')
    (ROOT/'interpretations.md').write_text('\n'.join('- '+s for s in story),encoding='utf-8')

if __name__=='__main__': main()
