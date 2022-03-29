import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go
import plotly_express as px
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_curve,auc
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.multiclass import OneVsRestClassifier
import streamlit.components.v1 as components
# Load our pkgs
import lime
import lime.lime_tabular
import numpy as np
import random
import eli5
from eli5 import show_weights
from eli5.sklearn import PermutationImportance
from eli5 import show_prediction
from eli5.sklearn import explain_decision_tree
from eli5.sklearn import explain_weights_sklearn
# run block of code and catch warnings
import warnings
warnings.filterwarnings("ignore")
	# execute code that will generate warnings

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    background = "<style>:root {background-color: white;}</style>"
    #components.html(yellow_background + html_content)
    components.html(background+shap_html, height=height)
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

def plot(df,Y):
    st.subheader('*Data Description*')
    st.write(df.describe())
    st.subheader('*Data Correlation*')
  
    corr = df.corr(method ='pearson')
    fig_corr = plt.figure(figsize=(12, 4))
    sns.heatmap(corr)
    st.pyplot(fig_corr)
    st.subheader('*Data presentation*')
    fig_desc = sns.pairplot(df, hue=Y.name)
    st.pyplot(fig_desc)
    Scatterplot(df,Y)
#---------------------------------#
# Model building
@st.cache(suppress_st_warning=True)
def build_model(df,target,model):
    # create label encoder object

    if target=='':
        X = df.iloc[ : , :-1]
        Y = df.iloc[ : , -1]
    else:

        X = df.drop([target], axis=1) # Using all column except for the last column as X
        Y = df[target] # Selecting the last column as Y

    plot(df,Y)

    if model=='Logistic Regression':
        model= LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=10000,multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    elif model=='Random Forest Regressor':
        model=RandomForestRegressor()
    elif model=='Linear Regression​':
        model= LinearRegression()
    elif model=='Decision Tree Regressor​':
        model= DecisionTreeRegressor()
    elif model=='Support Vector Regression':
        model= SVR(kernel='linear')
    elif model=='RandomForestClassifier':
        model=RandomForestClassifier()
    else:
        model=  OneVsRestClassifier(LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=10000,multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None))    



    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
    st.subheader('*Quality of Training and Testing (QoT) Toolkit*')
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)
    
    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

   
    model.fit(X_train, Y_train)
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc={}
    n_class = len(Y.unique())
    st.subheader('2. Model Performance')
    st.markdown('**Feature importance**')
    name=type(model).__name__
    st.write(name)

    if name=='LogisticRegression' or name=='SVR'or name=='OneVsRestClassifier':  
        st.info(model.coef_[0])
        col_sorted_by_importance=model.coef_[0].argsort()
        feat_imp=pd.DataFrame({
        'cols':X.columns[col_sorted_by_importance],
        'imps':model.coef_[0][col_sorted_by_importance]
        })
    elif name=='DecisionTreeRegressor' or name=='RandomForestRegressor':          
        st.info(model.feature_importances_)
        col_sorted_by_importance=model.feature_importances_.argsort()
        feat_imp=pd.DataFrame({
        'cols':X.columns[col_sorted_by_importance],
        'imps':model.feature_importances_[col_sorted_by_importance]
        })
    else:          
        st.info(model.coef_)
        col_sorted_by_importance=model.coef_.argsort()
        feat_imp=pd.DataFrame({
        'cols':X.columns[col_sorted_by_importance],
        'imps':model.coef_[col_sorted_by_importance]
        })

    #Axis to color
    color="imps", 
    fig = px.bar(        
        feat_imp,
        x = "imps",
        y = "cols",
        title = "Feauture importance",
        color="imps",
        orientation = 'h'
    )
    
    st.plotly_chart(fig)
    st.markdown('**2. Accuracy**')
    pred = model.predict(X_test)
    #st.write(pred)
    #st.write(Y_test)

    fig = plt.figure(figsize=(12, 4))
    ax1 = sns.distplot(Y_test, hist=False, color="r", label="Actual Value")
    sns.distplot(pred, hist=False, color="b", label="Fitted Values" , ax=ax1)
    fig.legend(labels=['Actual','Predicted'])
    st.pyplot(fig)
    acc=0
   

       
    if (name=='LogisticRegression'or name=='OneVsRestClassifier'):
       acc=accuracy_score(pred, Y_test)*100
    else : 
    # Performance metrics
        errors = abs(pred - Y_test)
        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / Y_test)
        # Calculate and display accuracy
        #acc = 100 - np.mean(mape)
        acc = model.score(X_test, Y_test)*100

    st.info(acc)
    notguessed=100-acc
    text= str("{:.2f}".format(acc))+ '% Accuracy'
    fig2 = go.Figure(data=[go.Pie( values=[acc,notguessed], pull=[0, 0.2],marker_colors=['blue','#0e1117'], textinfo='none')],layout =go.Layout( {"showlegend": False}))
    fig2.update_traces(hole=.8, hoverinfo="percent")
    fig2.add_annotation(x= 0.5, y = 0.5,
                    text = text,
                    font = dict(size=20,family='Verdana', 
                                color='white'),
                    showarrow = False)
    st.plotly_chart(fig2)




    st.markdown('**2.1. Training set**')
    Y_pred_train = model.predict(X_train)
    

    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = model.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    if name=='LogisticRegression' or name=='OneVsRestClassifier':
        cl=classification_report(Y_test, pred)
        cm=confusion_matrix(Y_test, pred)
        fig6=plt.figure(figsize=(12, 6))
        ax2=sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Accuracy Score:{:.2f}'.format(acc)
        plt.title(all_sample_title)
        col1, col2 = st.columns(2)
        col1.subheader("Classification report ")
        col1.text(cl)
        col2.subheader("Confusion Matrix")
        col2.write(fig6)
        pred_prob = model.predict_proba(X_test)
        from random import randint
        colors = []
        precision={}
        recall={}
        for i in range(10):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        for i in range(n_class):    
            fpr[i], tpr[i], thresh[i] = roc_curve(Y_test, pred_prob[:,i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])  
        for x in range(n_class):   
            precision[x], recall[x], _ = precision_recall_curve(Y_test,pred_prob[:,x], pos_label=x)                         
                                            
        fig7=plt.figure(figsize=(12, 6))
        for y in range(n_class):        
            plt.plot(fpr[y], tpr[y], linestyle='--',color=colors[y], label= 'Class:{0}'.format(y))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
    
        #plt.show()
        #plt.show()
        #precision, recall, _ = metrics.precision_recall_curve(Y_test, pred,pos_label=1)
        #disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        #disp.plot()
    # plt.show()

        col3, col4 = st.columns(2)
        col3.subheader("ROC Curve Plot for all classes")

        col3.pyplot( fig7)

        col4.subheader("ROC Curve Plot for selected class")
        
        sel = col4.selectbox('Select class:',Y.unique())
        col4.pyplot(Roc_curve(fpr, tpr, Y,colors,sel))
        fig10=plt.figure(figsize=(12, 6))
        for z in range(n_class): 
            plt.plot(recall[z], precision[z], lw=2, label='Class {0}'.format(z))

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")
        #plt.show()
        
        st.header("PR AUC Plot")

        st.pyplot( fig10)       
    
    st.subheader('3. Model Parameters')
    st.write(model.get_params())
    
    st.subheader('*Explainable AI*')
    #st.header('')
    with st.expander("Feature Contribution to Model Output"):    
        Shap(model,X_test,X_train,Y)
    with st.expander("White Box Model"):
        Lime(X_train,Y_test,X_test,model)
    with st.expander("Black Box Model Inspector"):
        if name=='LogisticRegression' or name=='OneVsRestClassifier':
            perm = PermutationImportance(model, scoring = 'accuracy' ,random_state=101).fit(X_test, Y_test)

        #show_weights(perm, feature_names = list(X_test.columns))
        
        background = "<style>:root {background-color: white;}</style>"
        if name=='LogisticRegression' or name=='OneVsRestClassifier':
            html_object =show_weights(perm,feature_names=list(X_test.columns),target_names=Y_test)
        else:
            html_object =show_weights(model, feature_names=list(X_test.columns),target_names=Y_test)
        raw_html = html_object._repr_html_()
        components.html(background+raw_html,height=300,scrolling=True)
        #with st.container():
        if name=='LogisticRegression' or name=='OneVsRestClassifier':
            html_object=show_prediction(model,X_test.iloc[0], feature_names=list(X_test.columns),target_names=Y.unique, show_feature_values=True)        
        else:    
            html_object=show_prediction(model, X_test.iloc[0], feature_names=list(X_test.columns),target_names=Y.unique, show_feature_values=True)
        raw_html = html_object._repr_html_()
        components.html(background+raw_html,height=300,scrolling=True)

        html_object=eli5.explain_weights(model)
       
        raw_html = html_object._repr_html_()
        components.html(background+raw_html,height=300,scrolling=True)
        html_object=eli5.explain_prediction(model, X.head(1))
        raw_html = html_object._repr_html_()
        components.html(background+raw_html,height=300,scrolling=True)

#Scatterplot(df, Y)
def Lime(X_train,y_test,X_test,model):
    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),mode="regression", feature_names=list(X_train.columns), class_names=y_test, discretize_continuous=True)
    # The Explainer Instance
    idx = random.randint(1, len(X_test))
    exp = explainer.explain_instance(X_test.iloc[3], model.predict, top_labels=1)
    #html=exp.as_pyplot_figure()
    exp.show_in_notebook(show_table=True)
    html = exp.as_html()
    background = "<style>:root {background-color: white;}</style>"
    #components.html(yellow_background + html_content)
    components.html(background+html, height=300,scrolling=True)    
    st.write(exp.as_list())
    #st.title("Newsgroup Classifier")
    #st.write(f"Document id = {idx}")
    #st.write(f"Probability(christian) = {c.predict_proba([newsgroups_test.data[idx]])[0,1]}")
    #st.write(f"True class:  {class_names[newsgroups_test.target[idx]]}")
    #exp.as_pyplot_figure()
    #st.pyplot()
    #plt.clf()
    #st.markdown(exp.as_html(), unsafe_allow_html=True)

def Scatterplot(df,target):
    selected_x_var = st.selectbox('What do you want the x variable to be?', df.columns)
    selected_y_var = st.selectbox('What about the y?', df.columns)
    fig5 = px.scatter(df, x = df[selected_x_var], y = df[selected_y_var], color=target)
    st.plotly_chart(fig5)

def Shap(model,X_test,X_train,Y):
    shap.initjs()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    name=type(model).__name__
    
    if name=='LogisticRegression' or name=='OneVsRestClassifier':
        explainer = shap.KernelExplainer(model.predict_proba, X_train, feature_names=X_train.columns)
        shap_values = explainer.shap_values(X_test)
        
        plt.title('Feature importance based on SHAP values')
        fig11=plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values,X_test,feature_names=X_train.columns)    
        st.pyplot (fig11)
        st.write('---')
    
        unique=Y.unique()
        unique
        for ind in range(len(unique)):
            st.subheader('Class -{0}'.format(unique[ind]))
            st_shap(shap.force_plot(explainer.expected_value[ind], shap_values[ind], X_test,feature_names=X_train.columns),500)
            st.write('---')

        for ind in range(len(unique)):
            st.subheader('Class -{0}'.format(unique[ind]))
            fig12=plt.figure(figsize=(12, 6))
            shap.summary_plot(shap_values[ind],X_test,feature_names=X_train.columns,class_names=unique)
            st.pyplot(fig12)
            st.write('---')

        for i in range(len(unique)):
            ind = i
            st.write(X_test.iloc[ind])
            #st.write(explainer.expected_value[ind])    
            shap_display=shap.force_plot(explainer.expected_value[ind],shap_values[ind][0],X_test.iloc[ind],feature_names=X_train.columns)
            st_shap(shap_display,150) 
            st.write('---')
    else:
        rf_explainer = shap.KernelExplainer(model.predict,X_test, feature_names=X_test.columns)
        shap_values=rf_explainer.shap_values(X_test)
        plt.title('Feature importance based on SHAP values')
        fig20=plt.figure()
        shap.summary_plot(shap_values, X_test)
        #fig20.savefig("/summary_plot1.png", bbox_inches='tight', dpi=600)
        st.pyplot (fig20)
        st.write('---')
        st.write(X_test.iloc[10,:])
        
        st_shap(shap.force_plot(rf_explainer.expected_value, shap_values[10,:], X_test.iloc[10,:]))
        
        st.write('---')
        st_shap(shap.force_plot(rf_explainer.expected_value, shap_values, X_test),500)
        st.write('---')
'''''        
'            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train, approximate=False, check_additivity=False)
            plt.title('Feature importance based on SHAP values')
            fig20=plt.figure(figsize=(12, 6))
            shap.summary_plot(shap_values, X_train)
            st.pyplot (fig20)
            st.write('---')'
'''''

def Roc_curve(fpr,tpr,Y,colors,sel):
    fig8=plt.figure(figsize=(12, 6))
    #sel = st.selectbox('Select class:',Y.unique())
    plt.plot(fpr[sel], tpr[sel], linestyle='--',color='red', label= 'Class:{0}'.format(sel))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    return fig8
