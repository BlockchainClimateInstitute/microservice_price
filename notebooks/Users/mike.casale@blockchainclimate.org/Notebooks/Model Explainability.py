# Databricks notebook source
# MAGIC %md
# MAGIC # Model Explainability
# MAGIC 
# MAGIC #### Explaining BCI AVM's XGBoost Tree Model with Path-Dependent Feature Perturbation using Tree SHAP
# MAGIC 
# MAGIC #### About SHAP Values
# MAGIC 
# MAGIC SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions (see papers for details and citations).
# MAGIC 
# MAGIC 
# MAGIC <p align="center">
# MAGIC <img width=25% src="https://blockchainclimate.org/wp-content/uploads/2020/11/cropped-BCI_Logo_LR-400x333.png" alt="bciAVM" height="300"/>
# MAGIC </p>
# MAGIC 
# MAGIC [![PyPI](https://badge.fury.io/py/bciavm.svg?maxAge=2592000)](https://badge.fury.io/py/bciavm)
# MAGIC [![PyPI Stats](https://img.shields.io/badge/bciavm-avm-blue)](https://pypistats.org/packages/bciavm)
# MAGIC 
# MAGIC 
# MAGIC This notebook contains code to take a `mlflow` registered model and distribute its work with a `Dask` cluster. 
# MAGIC <table>
# MAGIC     <tr>
# MAGIC         <td>
# MAGIC             <img width=25% src="https://saturn-public-assets.s3.us-east-2.amazonaws.com/example-resources/dask.png" width="300">
# MAGIC         </td>
# MAGIC     </tr>
# MAGIC </table>
# MAGIC 
# MAGIC The Blockchain & Climate Institute (BCI) is a progressive think tank providing leading expertise in the deployment of emerging technologies for climate and sustainability actions. 
# MAGIC 
# MAGIC As an international network of scientific and technological experts, BCI is at the forefront of innovative efforts, enabling technology transfers, to create a sustainable and clean global future.
# MAGIC 
# MAGIC # Automated Valuation Model (AVM) 
# MAGIC 
# MAGIC ### About
# MAGIC AVM is a term for a service that uses mathematical modeling combined with databases of existing properties and transactions to calculate real estate values. 
# MAGIC The majority of automated valuation models (AVMs) compare the values of similar properties at the same point in time. 
# MAGIC Many appraisers, and even Wall Street institutions, use this type of model to value residential properties. (see [What is an AVM](https://www.investopedia.com/terms/a/automated-valuation-model.asp) Investopedia.com)
# MAGIC 
# MAGIC For more detailed info about the AVM, please read the **About** paper found here `resources/2021-BCI-AVM-About.pdf`.
# MAGIC 
# MAGIC ### Valuation Process
# MAGIC <img src="resources/valuation_process.png" height="360" >
# MAGIC 
# MAGIC **Key Functionality**
# MAGIC 
# MAGIC * **Supervised algorithms** 
# MAGIC * **Tree-based & deep learning algorithms** 
# MAGIC * **Feature engineering derived from small clusters of similar properties** 
# MAGIC * **Ensemble (value blending) approaches** 
# MAGIC 
# MAGIC ### Set the required AWS Environment Variables
# MAGIC ```shell
# MAGIC export ACCESS_KEY=YOURACCESS_KEY
# MAGIC export SECRET_KEY=YOURSECRET_KEY
# MAGIC export BUCKET_NAME=bci-transition-risk-data
# MAGIC export TABLE_DIRECTORY=/dbfs/FileStore/tables/
# MAGIC ```
# MAGIC 
# MAGIC ### Next Steps
# MAGIC Read more about bciAVM on our [documentation page](https://blockchainclimate.org/thought-leadership/#blog):
# MAGIC 
# MAGIC ### How does it relate to BCI Risk Modeling?
# MAGIC <img src="resources/bci_flowchart_2.png" height="280" >
# MAGIC 
# MAGIC 
# MAGIC ### Technical & financial support for development provided by:
# MAGIC <a href="https://www.gcode.ai">
# MAGIC     <img width=15% src="https://staticfiles-img.s3.amazonaws.com/avm/gcode_logo.png" alt="GCODE.ai"  height="25"/>
# MAGIC </a>
# MAGIC 
# MAGIC 
# MAGIC ### Install [from PyPI](https://pypi.org/project/bciavm/)
# MAGIC ```shell
# MAGIC pip install bciavm
# MAGIC ```
# MAGIC 
# MAGIC This notebook covers the following steps:
# MAGIC - Import data from your local machine into the Databricks File System (DBFS)
# MAGIC - Download data from s3
# MAGIC - Train a machine learning models (or more technically, multiple models in a stacked pipeline) on the dataset
# MAGIC - Register the model in MLflow

# COMMAND ----------

import bciavm, shap
import numpy as np
import pandas as pd
from functools import partial

x_path, y_path = '/dbfs/FileStore/tables/avm/X_test.csv', '/dbfs/FileStore/tables/avm/y_test.csv'
pipe_path = '/dbfs/FileStore/artifacts/avm_pipeline_'+str(bciavm.__version__)+'.pkl'
target = 'Price_p'
data_index = 'unit_indx'

# COMMAND ----------

pipeline = bciavm.pipelines.RegressionPipeline.load(pipe_path)
model = pipeline.get_component('XGB Regressor')._component_obj
model

# COMMAND ----------

#the avm model features
features = pipeline.get_component('XGB Regressor').input_feature_names
print('# of features = ', len(features))
features

# COMMAND ----------

X = pd.read_csv(x_path)
y = pd.read_csv(y_path)

#create the explainer
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

# define charts
def _dependence_plot(features, shap_values, dataset, feature_names, fig_height=15, fig_width=10,  display_features=None, **kwargs):
    """ 
    Plots dependence plots of specified features in a grid.
    
    features: List[str], List[Tuple[str, str]]
        Names of features to be plotted. If List[str], then shap 
        values are plotted as a function of feature value, coloured 
        by the value of the feature determined to have the strongest
        interaction (empirically). If List[Tuple[str, str]], shap
        interaction values are plotted.
    display_features: np.ndarray, N x F
        Same as dataset, but contains human readable values
        for categorical levels as opposed to numerical values
    """
    
    def _set_fonts(fig, ax, fonts=None, set_cbar=False):
        """
        Sets fonts for axis labels and colobar.
        """

        ax.xaxis.label.set_size(xlabelfontsize)
        ax.yaxis.label.set_size(ylabelfontsize)
        ax.tick_params(axis='x', labelsize=xtickfontsize)
        ax.tick_params(axis='y', labelsize=ytickfontsize)
        if set_cbar:
            fig.axes[-1].tick_params(labelsize=cbartickfontsize)
            fig.axes[-1].tick_params(labelrotation=cbartickrotation)
            fig.axes[-1].yaxis.label.set_size(cbarlabelfontsize)

    # parse plotting args
    figsize = kwargs.get("figsize", (fig_height, fig_width))
    nrows = kwargs.get('nrows', len(features))
    ncols = kwargs.get('ncols', 1)
    xlabelfontsize = kwargs.get('xlabelfontsize', 14)
    xtickfontsize = kwargs.get('xtickfontsize', 11)
    ylabelfontsize = kwargs.get('ylabelfontsize', 14)
    ytickfontsize = kwargs.get('ytickfontsize', 11)
    cbartickfontsize = kwargs.get('cbartickfontsize', 14)
    cbartickrotation = kwargs.get('cbartickrotation', 45)
    cbarlabelfontsize = kwargs.get('cbarlabelfontsize', 14)
    rotation_orig = kwargs.get('xticklabelrotation', 25)
    alpha = kwargs.get("alpha", 1)
    x_jitter_orig = kwargs.get("x_jitter", 0.8)
    grouped_features = list(zip_longest(*[iter(features)] * ncols))
    
    fig, axes = plt.subplots(nrows, ncols,  figsize=figsize)
    if nrows == len(features):
        axes = list(zip_longest(*[iter(axes)] * 1))


    for i, (row, group) in enumerate(zip(axes, grouped_features), start=1):
        # plot each feature or interaction in a subplot
        for ax, feature in zip(row, group):
            # set x-axis ticks and labels and x-jitter for categorical variables
            if not feature:
                continue
            if isinstance(feature, list) or isinstance(feature, tuple):
                feature_index = feature_names.index(feature[0])
            else:
                feature_index = feature_names.index(feature)

            x_jitter = 0
            
            shap.dependence_plot(feature, 
                                 shap_values,
                                 dataset,
                                 feature_names=feature_names,
                                 display_features=display_features,
                                 interaction_index='auto',
                                 ax=ax,
                                 show=False,
                                 x_jitter=x_jitter,
                                 alpha=alpha
                                )
            if i!= nrows:
                ax.tick_params('x', labelrotation=rotation_orig)
            _set_fonts(fig, ax, set_cbar=True)
    

def plot_decomposition(feature_pair, shap_interaction_vals, features, feat_names, display_features=None, **kwargs):
    """
    Given a list containing two feature names (`feature_pair`), an n_instances x n_features x n_features tensor 
    of shap interaction values (`shap_interaction_vals`), an n_instances x n_features (`features`) tensor of 
    feature values and a list of feature names (which assigns a name to each column of `features`), this function 
    plots:
        - left: shap values for feature_pair[0] coloured by the value of feature_pair[1]
        - middle: shap values for feature_pair[0] after subtracting the interaction with feature_pair[1]
        - right: the interaction values between feature_pair[0] and feature_pair[1], which are subtracted 
        from the left plot to get the middle plot
        
    NB: `display_features` is the same shape as `features` but should contain the raw categories for categorical 
    variables so that the colorbar can be discretised and the category names displayed alongside the colorbar.
    """
    
    def _set_fonts(fig, ax, fonts=None, set_cbar=False):
        """
        Sets fonts for axis labels and colobar.
        """

        ax.xaxis.label.set_size(xlabelfontsize)
        ax.yaxis.label.set_size(ylabelfontsize)
        ax.tick_params(axis='x', labelsize=xtickfontsize)
        ax.tick_params(axis='y', labelsize=ytickfontsize)
        if set_cbar:
            fig.axes[-1].tick_params(labelsize=cbartickfontsize)
            fig.axes[-1].yaxis.label.set_size(cbarlabelfontsize)

    # parse plotting args
    xlabelfontsize = kwargs.get('xlabelfontsize', 21)
    ylabelfontsize = kwargs.get('ylabelfontsize', 16)
    cbartickfontsize = kwargs.get('cbartickfontsize', 16)
    cbarlabelfontsize = kwargs.get('cbarlabelfontsize', 21)
    xtickfontsize = kwargs.get('xtickfontsize', 20)
    ytickfontsize = kwargs.get('ytickfontsize', 16)
    alpha = kwargs.get('alpha', 0.7)
    figsize = kwargs.get('figsize', (44, 16))
    ncols = kwargs.get('ncols', 3)
    nrows = kwargs.get('nrows', 1)
    
    # compute shap values and shap values without interaction
    feat1_idx = feat_names.index(feature_pair[0])
    feat2_idx = feat_names.index(feature_pair[1])
    
    # shap values
    shap_vals = shap_interaction_vals.sum(axis=2)
    
    # shap values for feat1, all samples
    shap_val_ind1 = shap_interaction_vals[..., feat1_idx].sum(axis=1)
    
    # shap values for (feat1, feat2) interaction 
    shap_int_ind1_ind2 = shap_interaction_vals[:, feat2_idx, feat1_idx]
    
    # subtract effect of feat2
    shap_val_minus_ind2 = shap_val_ind1 - shap_int_ind1_ind2
    shap_val_minus_ind2 = shap_val_minus_ind2[:, None]

    # create plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows, ncols, figsize=figsize)

    # plot the shap values including the interaction
    shap.dependence_plot(feature_pair[0],
                         shap_vals,
                         features,
                         display_features = display_features,
                         feature_names=feat_names,
                         interaction_index=feature_pair[1],
                         alpha=alpha,
                         ax=ax1,
                         show=False)
    _set_fonts(fig, ax1, set_cbar=True)

    # plot the shap values excluding the interaction
    shap.dependence_plot(0,
                         shap_val_minus_ind2,
                         features[:, feat1_idx][:, None],
                         feature_names=[feature_pair[0]],
                         interaction_index=None,
                         alpha=alpha,
                         ax=ax2,
                         show=False,
                         )
    ax2.set_ylabel(f' Shap value for  {feature_pair[0]} \n wo {feature_pair[1]} interaction')
    _set_fonts(fig, ax2)
    
    # plot the interaction value
    shap.dependence_plot(feature_pair,
                         shap_interaction_vals,
                         features,
                         feature_names=feat_names,
                         display_features=display_features,
                         interaction_index='auto',
                         alpha=alpha,
                         ax=ax3,
                         show=False,
                        )
    _set_fonts(fig, ax3, set_cbar=True)
    



def plot_decomposition(feature_pair, shap_interaction_vals, features, feat_names, display_features=None, **kwargs):
    """
    Given a list containing two feature names (`feature_pair`), an n_instances x n_features x n_features tensor 
    of shap interaction values (`shap_interaction_vals`), an n_instances x n_features (`features`) tensor of 
    feature values and a list of feature names (which assigns a name to each column of `features`), this function 
    plots:
        - left: shap values for feature_pair[0] coloured by the value of feature_pair[1]
        - middle: shap values for feature_pair[0] after subtracting the interaction with feature_pair[1]
        - right: the interaction values between feature_pair[0] and feature_pair[1], which are subtracted 
        from the left plot to get the middle plot
        
    NB: `display_features` is the same shape as `features` but should contain the raw categories for categorical 
    variables so that the colorbar can be discretised and the category names displayed alongside the colorbar.
    """
    
    def _set_fonts(fig, ax, fonts=None, set_cbar=False):
        """
        Sets fonts for axis labels and colobar.
        """

        ax.xaxis.label.set_size(xlabelfontsize)
        ax.yaxis.label.set_size(ylabelfontsize)
        ax.tick_params(axis='x', labelsize=xtickfontsize)
        ax.tick_params(axis='y', labelsize=ytickfontsize)
        if set_cbar:
            fig.axes[-1].tick_params(labelsize=cbartickfontsize)
            fig.axes[-1].yaxis.label.set_size(cbarlabelfontsize)

    # parse plotting args
    xlabelfontsize = kwargs.get('xlabelfontsize', 21)
    ylabelfontsize = kwargs.get('ylabelfontsize', 16)
    cbartickfontsize = kwargs.get('cbartickfontsize', 16)
    cbarlabelfontsize = kwargs.get('cbarlabelfontsize', 21)
    xtickfontsize = kwargs.get('xtickfontsize', 20)
    ytickfontsize = kwargs.get('ytickfontsize', 16)
    alpha = kwargs.get('alpha', 0.7)
    figsize = kwargs.get('figsize', (44, 16))
    ncols = kwargs.get('ncols', 3)
    nrows = kwargs.get('nrows', 1)
    # compute shap values and shap values without interaction
    feat1_idx = feat_names.index(feature_pair[0])
    feat2_idx = feat_names.index(feature_pair[1])
    # shap values
    shap_vals = shap_interaction_vals.sum(axis=2)
    # shap values for feat1, all samples
    shap_val_ind1 = shap_interaction_vals[..., feat1_idx].sum(axis=1)
    # shap values for (feat1, feat2) interaction 
    shap_int_ind1_ind2 = shap_interaction_vals[:, feat2_idx, feat1_idx]
    # subtract effect of feat2
    shap_val_minus_ind2 = shap_val_ind1 - shap_int_ind1_ind2
    shap_val_minus_ind2 = shap_val_minus_ind2[:, None]

    # create plot

    fig, (ax1, ax2, ax3) = plt.subplots(nrows, ncols, figsize=figsize)

    # plot the shap values including the interaction
    shap.dependence_plot(feature_pair[0],
                         shap_vals,
                         features,
                         display_features = display_features,
                         feature_names=feat_names,
                         interaction_index=feature_pair[1],
                         alpha=alpha,
                         ax=ax1,
                         show=False)
    _set_fonts(fig, ax1, set_cbar=True)

    # plot the shap values excluding the interaction
    shap.dependence_plot(0,
                         shap_val_minus_ind2,
                         features[:, feat1_idx][:, None],
                         feature_names=[feature_pair[0]],
                         interaction_index=None,
                         alpha=alpha,
                         ax=ax2,
                         show=False,
                         )
    ax2.set_ylabel(f' Shap value for  {feature_pair[0]} \n wo {feature_pair[1]} interaction')
    _set_fonts(fig, ax2)
    
    # plot the interaction value
    shap.dependence_plot(feature_pair,
                         shap_interaction_vals,
                         features,
                         feature_names=feat_names,
                         display_features=display_features,
                         interaction_index='auto',
                         alpha=alpha,
                         ax=ax3,
                         show=False)
    
    _set_fonts(fig, ax3, set_cbar=True)
    


# COMMAND ----------

preprocessor = pipeline.get_component("Preprocess Transformer")
imputer = pipeline.get_component('Imputer')
ohe = pipeline.get_component('One Hot Encoder')

# COMMAND ----------

X_processed = X.copy()
X_t = preprocessor.transform(X_processed)
X_t = imputer.transform(X_t)
X_t = ohe.transform(X_t)
X_t = X_t[features]
X_t

# COMMAND ----------

#Use a small sample bc SHAP is slow
Xs = X_t.sample(100)

# #calculate the shap values
shap_values = explainer.shap_values(Xs.values, check_additivity=False)

# #calculate the interaction values
shap_interaction_values = explainer.shap_interaction_values(Xs.values)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 1: Features that lower the AVM valuation
# MAGIC - A lower `area` feature is generally indicative of a lower AVM valuation
# MAGIC - A lower `comparables_75_sale_price` feature is generally indicative of a lower AVM valuation
# MAGIC - A lower `floor_number` is generally indicative of a lower AVM value
# MAGIC - A lower `plot_size` or a *NULL (grey area)* `plot_size` is generally indicative of a lower AVM valuation
# MAGIC - A lower `cam_fee` is generally indicative of a lower AVM value
# MAGIC - A *False* `is_allow_foreign_owner` is generally indicative of a lower AVM value
# MAGIC - A `month` that is *later in the year*, denoted by a higher `month` value, is generally indicative of a lower AVM value

# COMMAND ----------

shap.summary_plot(shap_values, Xs, features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 2: Features that increase the AVM valuation
# MAGIC - A higher `area` feature is generally indicative of a higher AVM valuation
# MAGIC - A higher `comparables_75_sale_price` feature is generally indicative of a higher AVM valuation
# MAGIC - A higher `floor_number` is generally indicative of a higher AVM value
# MAGIC - A higher `plot_size` is generally indicative of a higher AVM valuation
# MAGIC - A higher `cam_fee` is generally indicative of a higher AVM value
# MAGIC - A *True* `is_allow_foreign_owner` is generally indicative of a higher AVM value
# MAGIC - A `month` that is *earlier in the year*, denoted by a lower `month` value, is generally indicative of a higher AVM value

# COMMAND ----------

shap.summary_plot(shap_values, Xs, features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 3: Interesting Partial Dependence Plots
# MAGIC 
# MAGIC While the above SHAP summary plots give a general overview of each feature, the below SHAP dependence plots show how the model output varies by feauture value. 
# MAGIC 
# MAGIC In the Summary Plots, note that every dot is a `unit_indx` valuation, and the vertical dispersion at a single feature value results from interaction effects in the model. The feature used for coloring is automatically chosen to highlight what might be driving these interactions. Later we will see how to check that the interaction is really in the model with SHAP interaction values. Note that the row of a SHAP summary plot results from projecting the points of a SHAP dependence plot onto the y-axis, then recoloring by the feature itself.
# MAGIC 
# MAGIC Below we give the SHAP dependence plot for each of the chosen features, revealing interesting but expected trends. Keep in mind the calibration of some of these values can be different than a real world effect, so it is wise to be careful drawing concrete conclusions.
# MAGIC 
# MAGIC - **(top left)** 
# MAGIC 
# MAGIC - **(top right)** 
# MAGIC 
# MAGIC - **(bottom left)** 
# MAGIC 
# MAGIC - **(bottom right)** 

# COMMAND ----------

from itertools import product, zip_longest
import matplotlib.pyplot as plt

plot_dependence = partial(
    _dependence_plot, 
    feature_names=features,
    category_map={},
)

plot_dependence(
    [ 
     'TOTAL_FLOOR_AREA_e',
     'NUMBER_HEATED_ROOMS_e', 
     'FLOOR_LEVEL_e', 
     'Price_p__median'
    ], 
    shap_values, 
    Xs, 
    display_features = Xs,  
    rotation=33,
    figsize=(40, 20), 
    alpha=1, 
    x_jitter=0.5,
    nrows=2,
    ncols=2,
    xlabelfontsize=24,
    xtickfontsize=20,
    xticklabelrotation=0,
    ylabelfontsize=24,
    ytickfontsize=21,
    cbarlabelfontsize=22,
    cbartickfontsize=20,
    cbartickrotation=0
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Figure 4: Percent Error By Price Range
# MAGIC 
# MAGIC We can calculate and graph the % Error by Price Range as follows:

# COMMAND ----------


def _plot_accuracy_by_price_range(y_test, y_stacked, units):
    model_name='avm'

    df_acc = pd.DataFrame({'y_test': y_test, 'preds': y_stacked, 'unit_indx': units}, columns=['y_test', 'preds', 'unit_indx'])
    df_acc['error'] = abs((df_acc['preds'] - df_acc['y_test']) / df_acc['y_test'])
    df_acc['percent_error'] = (df_acc['preds'] - df_acc['y_test']) / df_acc['y_test']

    ctt = []
    try: 
        if len(df_acc[df_acc['y_test']<=50000]) > 5: ctt.append('df_0')
    except: pass
    try: 
        if len(df_acc[(df_acc['y_test']>=50000) & (df_acc['y_test']<=100000)]) > 5: ctt.append('df_5')
    except: pass
    try: 
        if len(df_acc[(df_acc['y_test']>=100000) & (df_acc['y_test']<=150000)]) > 5: ctt.append('df_10')
    except: pass
    try: 
        if len(df_acc[(df_acc['y_test']>=150000) & (df_acc['y_test']<=200000)]) > 5: ctt.append('df_15')
    except: pass
    try: 
        if len(df_acc[(df_acc['y_test']>=200000) & (df_acc['y_test']<=250000)]) > 5: ctt.append('df_20')
    except: pass
    try: 
        if len(df_acc[(df_acc['y_test']>=250000) & (df_acc['y_test']<=300000)]) > 5: ctt.append('df_25')
    except: pass
    try: 
        if len(df_acc[(df_acc['y_test']>=300000) & (df_acc['y_test']<=350000)]) > 5: ctt.append('df_30')
    except: pass
    try: 
        if len(df_acc[(df_acc['y_test']>=350000) & (df_acc['y_test']<=400000)]) > 5: ctt.append('df_35')
    except: pass
    try: 
        if len(df_acc[df_acc['y_test']>=400000]) > 5: ctt.append('df_40')
    except: pass

    def troubleshoot(ctt, df_acc):

        acc_2 = len(ctt)*[0]
        acc_5 = len(ctt)*[0]
        acc_10 = len(ctt)*[0]
        acc_20 = len(ctt)*[0]

        x = 0
        if 'df_0' in ctt:
            try:
                df_0 = df_acc[df_acc['y_test']<=50000]
                acc_2[x] = len(df_0[df_0['error']<0.02])/len(df_0)
                acc_5[x] = len(df_0[df_0['error']<0.05])/len(df_0)
                acc_10[x] = len(df_0[df_0['error']<0.1])/len(df_0)
                acc_20[x] = len(df_0[df_0['error']<0.2])/len(df_0)
                x += 1
            except: 
                ctt.pop('df_0')
                troubleshoot(ctt, df_acc)

        if 'df_5' in ctt:
            try:
                df_5 = df_acc[(df_acc['y_test']>=50000) & (df_acc['y_test']<=100000)]
                acc_2[x] = len(df_5[df_5['error']<0.02])/len(df_5)
                acc_5[x] = len(df_5[df_5['error']<0.05])/len(df_5)
                acc_10[x] = len(df_5[df_5['error']<0.10])/len(df_5)
                acc_20[x] = len(df_5[df_5['error']<0.2])/len(df_5)
                x += 1
            except: 
                ctt.pop('df_5')
                troubleshoot(ctt, df_acc)

        if 'df_10' in ctt:
            try:
                df_10 = df_acc[(df_acc['y_test']>=100000) & (df_acc['y_test']<=150000)]
                acc_2[x] = len(df_10[df_10['error']<0.02])/len(df_10)
                acc_5[x] = len(df_10[df_10['error']<0.05])/len(df_10)
                acc_10[x] = len(df_10[df_10['error']<0.10])/len(df_10)
                acc_20[x] = len(df_10[df_10['error']<0.2])/len(df_10)
                x += 1
            except: 
                ctt.pop('df_10')
                troubleshoot(ctt, df_acc)

        if 'df_10' in ctt:
            try:
                df_15 = df_acc[(df_acc['y_test']>=150000) & (df_acc['y_test']<=200000)]
                acc_2[x] = len(df_15[df_15['error']<0.02])/len(df_15)
                acc_5[x] = len(df_15[df_15['error']<0.05])/len(df_15)
                acc_10[x] = len(df_15[df_15['error']<0.1])/len(df_15)
                acc_20[x] = len(df_15[df_15['error']<0.2])/len(df_15)
                x += 1
            except: 
                ctt.pop('df_10')
                troubleshoot(ctt, df_acc)

        if 'df_20' in ctt:
            try:
                df_20 = df_acc[(df_acc['y_test']>=200000) & (df_acc['y_test']<=250000)]
                acc_2[x] = len(df_20[df_20['error']<0.02])/len(df_20)
                acc_5[x] = len(df_20[df_20['error']<0.05])/len(df_20)
                acc_10[x] = len(df_20[df_20['error']<0.1])/len(df_20)
                acc_20[x] = len(df_20[df_20['error']<0.2])/len(df_20)
                x += 1
            except: 
                ctt.pop('df_20')
                troubleshoot(ctt, df_acc)

        if 'df_25' in ctt:
            try:
                df_25 = df_acc[(df_acc['y_test']>=250000) & (df_acc['y_test']<=300000)]
                acc_2[x] = len(df_25[df_25['error']<0.02])/len(df_25)
                acc_5[x] = len(df_25[df_25['error']<0.05])/len(df_25)
                acc_10[x] = len(df_25[df_25['error']<0.1])/len(df_25)
                acc_20[x] = len(df_25[df_25['error']<0.2])/len(df_25)
                x += 1
            except: 
                ctt.pop('df_25')
                troubleshoot(ctt, df_acc)

        if 'df_30' in ctt:
            try:
                df_30 = df_acc[(df_acc['y_test']>=300000) & (df_acc['y_test']<=350000)]
                acc_2[x] = len(df_30[df_30['error']<0.02])/len(df_30)
                acc_5[x] = len(df_30[df_30['error']<0.05])/len(df_30)
                acc_10[x] = len(df_30[df_30['error']<0.1])/len(df_30)
                acc_20[x] = len(df_30[df_30['error']<0.2])/len(df_30)
                x += 1
            except: 
                ctt.pop('df_30')
                troubleshoot(ctt, df_acc)

        if 'df_35' in ctt:
            try:
                df_35 = df_acc[(df_acc['y_test']>=350000) & (df_acc['y_test']<=400000)]
                acc_2[x] = len(df_35[df_35['error']<0.02])/len(df_35)
                acc_5[x] = len(df_35[df_35['error']<0.05])/len(df_35)
                acc_10[x] = len(df_35[df_35['error']<0.1])/len(df_35)
                acc_20[x] = len(df_35[df_35['error']<0.2])/len(df_35)
                x += 1
            except: 
                ctt.pop('df_35')
                troubleshoot(ctt, df_acc)

        if 'df_40' in ctt:
            try:
                df_40 = df_acc[df_acc['y_test']>=400000]
                acc_2[x] = len(df_40[df_40['error']<0.02])/len(df_40)
                acc_5[x] = len(df_40[df_40['error']<0.05])/len(df_40)
                acc_10[x] = len(df_40[df_40['error']<0.1])/len(df_40)
                acc_20[x] = len(df_40[df_40['error']<0.2])/len(df_40)
                x += 1
            except: 
                ctt.pop('df_40')
                troubleshoot(ctt, df_acc)

        return acc_2, acc_5, acc_10, acc_20, ctt

    acc_2, acc_5, acc_10, acc_20, ctt = troubleshoot(ctt, df_acc)

    acc = pd.DataFrame(
        {'acc_2': acc_2,
         'acc_5': acc_5,
         'acc_10': acc_10,
         'acc_20': acc_20,
        })

    a = []
    if 'df_0' in ctt:
        a.append('<50k')
    if 'df_5' in ctt:
        a.append('50k-100k')
    if 'df_10' in ctt:
        a.append('100k-150k')
    if 'df_15' in ctt:
        a.append('150k-200k')
    if 'df_20' in ctt:
        a.append('200k-250k')
    if 'df_25' in ctt:
        a.append('250k-300k')
    if 'df_30' in ctt:
        a.append('300k-350k')
    if 'df_35' in ctt:
        a.append('350k-400k')
    if 'df_40' in ctt:
        a.append('>400k')

    acc['value bands'] = a

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize = (20,10))
    #sns.catplot(x="acc_5", y="percentile", data=acc,ax=ax)
    ax.scatter(acc['acc_2'], acc['value bands'],label='err +-2%', s=80)
    ax.scatter(acc['acc_5'], acc['value bands'],label='err +-5%', s=80)
    ax.scatter(acc['acc_10'], acc['value bands'],label='err +-10%', s=80)
    ax.scatter(acc['acc_20'], acc['value bands'],label='err +-20%', s=80)

    for i in range(0, len(ctt)):
        plt.plot([acc['acc_2'][i], acc['acc_10'][i]], [[i]*5,[i]*5], 'grey')
        plt.plot([acc['acc_5'][i], acc['acc_20'][i]], [[i]*5,[i]*5], 'grey')

    plt.legend(fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_xticklabels([0,20,40,60,80,100], rotation=0, fontsize=15)
    ax.set_yticklabels(acc['value bands'], rotation=0, fontsize=15)
    ax.set_xlabel('Percent of valuations within +-5%, +-20%')
    ax.set_ylabel('Price range of properties £ Pounds')
    plt.title('Accuracy by Value ' + model_name,fontsize=20)
    plt.show()
    imname = 'Accuracy by Price Range.png'
    plt.savefig(imname)
    print('end plot_by_price_range...')
    
    X_misvalued_abs = df_acc[df_acc['error']< 0.2]
    X_misvalued_low = df_acc[df_acc['percent_error'] < -0.2]
    X_misvalued_high = df_acc[df_acc['percent_error'] > 0.2]
    return X_misvalued_low, X_misvalued_high, X_misvalued_abs, df_acc
    

# COMMAND ----------

Xx = X
y_true = y['Price_p'].values

# COMMAND ----------

y_predicted = pipeline.predict(Xx)

# COMMAND ----------

#sanity check the chart below

scores = pipeline.score(X, 
                        y['Price_p'], 
                        objectives=['MAPE',
                                 'MdAPE',
                                 'ExpVariance',
                                 'MaxError',
                                 'MedianAE',
                                 'MSE',
                                 'MAE',
                                 'R2',
                                 'Root Mean Squared Error'])
scores

# COMMAND ----------

# MAGIC %md
# MAGIC From the plot below we can see that in general:
# MAGIC - Approx 15% of valuations have an absolute error < +-2%
# MAGIC - Approx 35% of valuations have an absolute error < +-5%
# MAGIC - Over 55% of valuations have an absolute error < +-10%
# MAGIC - Approx 75% of valuations have an absolute error < +-20%

# COMMAND ----------

#plot and get high error units
X_misvalued_low, X_misvalued_high, X_misvalued_abs, df_acc = _plot_accuracy_by_price_range(y_true, y_predicted, Xx['unit_indx'].values)

# COMMAND ----------

#find high errors below the true value
X_misvalued_low_units = list(X_misvalued_low['unit_indx'].values)
X_misvalued_low = Xx[Xx['unit_indx'].isin(X_misvalued_low_units)]

#find high errors above the true value
X_misvalued_high_units = list(X_misvalued_high['unit_indx'].values)
X_misvalued_high = Xx[Xx['unit_indx'].isin(X_misvalued_high_units)]

#find high errors above the true value
X_misvalued_abs_units = list(X_misvalued_abs['unit_indx'].values)
X_misvalued_abs = Xx[Xx['unit_indx'].isin(X_misvalued_abs_units)]

columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
X_misvalued_low_t = preprocessor.transform(X_misvalued_low)
X_misvalued_low_t = imputer.transform(X_misvalued_low_t)
X_misvalued_low_t = ohe.transform(X_misvalued_low_t)
X_misvalued_low_t = X_misvalued_low_t[features]

X_misvalued_high_t = preprocessor.transform(X_misvalued_high)
X_misvalued_high_t = imputer.transform(X_misvalued_high_t)
X_misvalued_high_t = ohe.transform(X_misvalued_high_t)
X_misvalued_high_t = X_misvalued_high_t[features]

X_misvalued_abs_t = preprocessor.transform(X_misvalued_abs)
X_misvalued_abs_t = imputer.transform(X_misvalued_abs_t)
X_misvalued_abs_t = ohe.transform(X_misvalued_abs_t)
X_misvalued_abs_t = X_misvalued_abs_t[features]

X_misvalued_abs_t = X_misvalued_abs_t.sample(500)
X_misvalued_high_t = X_misvalued_high_t.sample(500)
X_misvalued_low_t = X_misvalued_low_t.sample(500)

#calc low & high error SHAP values
shap_values_misvalued_low = explainer.shap_values(X_misvalued_low_t.values, check_additivity=False)

# COMMAND ----------

shap_values_misvalued_high = explainer.shap_values(X_misvalued_high_t.values, check_additivity=False)

# COMMAND ----------

shap_values_misvalued_abs = explainer.shap_values(X_misvalued_abs_t.values, check_additivity=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 5: Understanding High Error Valuations
# MAGIC 
# MAGIC But what about valuations with over 20% error? Is there any way we can drill down on these valuations to understand what is happening, and perhaps make some corrections to fix, or at the very least, avoid providing valuations when we know the model is most likely to have a high error?
# MAGIC 
# MAGIC The good news is that we can in fact drill down on these high-error units. To do so, we first calculate the shap_values for only the high-error valuation units. The below SHAP Summary Plot is produced considering *only* high-error valuations. From the SHAP Summary Plot we can see:
# MAGIC 
# MAGIC - The feature that contributes most to high errors is `Price_p__median` which is an engineered (ie derived) feature built by looking at aggregate statistics about `comparable` units nearby the target property.
# MAGIC - In general, many of the engineered features play a large role in the high error valuations, but it's important to note that they also increase the preformance of the model overall (ie. without them, the model would perform worse). 
# MAGIC 
# MAGIC However, the overall model Summary Plots(s) and the high-error Summary Plot are very similar, and it's difficult to infer exactly what's happening from the Summary Plots alone.

# COMMAND ----------

shap.summary_plot(shap_values_misvalued_abs, X_misvalued_abs_t, features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 6: Insights - Absolute Error % 

# COMMAND ----------

plot_dependence = partial(
    _dependence_plot, 
    feature_names=features,
    category_map={},
)

plot_dependence(
    [ 
     'TOTAL_FLOOR_AREA_e',
     'NUMBER_HEATED_ROOMS_e', 
     'FLOOR_LEVEL_e', 
     'Price_p__median', 
     'Price_p__mean', 
     'TOTAL_FLOOR_AREA_e_minus_mean', 
     'NUMBER_HEATED_ROOMS_e_minus_mean',
     'Latitude_m', 
     'Longitude_m', 
     'FLOOR_LEVEL_e_minus_mean'
    ], 
    shap_values_misvalued_abs, 
    X_misvalued_abs_t,  
    display_features = X_misvalued_abs_t,  
    rotation=33,
    figsize=(47.5, 40), 
    alpha=1, 
    x_jitter=0.5,
    nrows=5,
    ncols=2,
    xlabelfontsize=24,
    xtickfontsize=20,
    xticklabelrotation=0,
    ylabelfontsize=24,
    ytickfontsize=21,
    cbarlabelfontsize=22,
    cbartickfontsize=20,
    cbartickrotation=0
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 7: High Error % Overestimation 
# MAGIC 
# MAGIC From the Summary Plot below we can see that the most important features influencing *overestimation* are very similary to the features influencing absolute error.

# COMMAND ----------

shap.summary_plot(shap_values_misvalued_high, X_misvalued_high_t, features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 8: Insights - Overestimation Error 
# MAGIC 
# MAGIC Some interesting patterns with `A` and `B` that can lead to **overestimation error** misvaluation are:
# MAGIC - a
# MAGIC - b
# MAGIC 
# MAGIC Some interesting patterns with `C` that can lead to **overestimation error** misvaluation are:
# MAGIC - c
# MAGIC 
# MAGIC Some interesting patterns with `D` that can lead to **overestimation error** misvaluation are:
# MAGIC - d

# COMMAND ----------

plot_dependence = partial(
    _dependence_plot, 
    feature_names=features,
    category_map={},
)

plot_dependence(
    [ 
     'TOTAL_FLOOR_AREA_e',
     'NUMBER_HEATED_ROOMS_e', 
     'FLOOR_LEVEL_e', 
     'Price_p__median', 
     'Price_p__mean', 
     'TOTAL_FLOOR_AREA_e_minus_mean', 
     'NUMBER_HEATED_ROOMS_e_minus_mean',
     'Latitude_m', 
     'Longitude_m', 
     'FLOOR_LEVEL_e_minus_mean'
    ], 
    shap_values_misvalued_high, 
    X_misvalued_high_t,  
    display_features = X_misvalued_high_t,  
    rotation=33,
    figsize=(47.5, 40), 
    alpha=1, 
    x_jitter=0.5,
    nrows=5,
    ncols=2,
    xlabelfontsize=24,
    xtickfontsize=20,
    xticklabelrotation=0,
    ylabelfontsize=24,
    ytickfontsize=21,
    cbarlabelfontsize=22,
    cbartickfontsize=20,
    cbartickrotation=0
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 9: Underestimation Error
# MAGIC 
# MAGIC From the Summary Plot below we can see that the most important features influencing *underestimation* are very similary to the features influencing absolute error.

# COMMAND ----------

shap.summary_plot(shap_values_misvalued_low, X_misvalued_low_t, features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 10: Insights - Underestimation Error 
# MAGIC 
# MAGIC Some interesting patterns with `A` that can lead to **underestimation error** misvaluation are:
# MAGIC - a
# MAGIC 
# MAGIC Some interesting patterns with `B` that can lead to **underestimation error** misvaluation are:
# MAGIC - b
# MAGIC 
# MAGIC Some interesting patterns with `C` that can lead to **underestimation error** misvaluation are:
# MAGIC - c
# MAGIC 
# MAGIC Some interesting patterns with `D` that can lead to **underestimation error** misvaluation are:
# MAGIC - d

# COMMAND ----------

plot_dependence = partial(
    _dependence_plot, 
    feature_names=features,
    category_map={},
)

plot_dependence(
    [ 
     'TOTAL_FLOOR_AREA_e',
     'NUMBER_HEATED_ROOMS_e', 
     'FLOOR_LEVEL_e', 
     'Price_p__median', 
     'Price_p__mean', 
     'TOTAL_FLOOR_AREA_e_minus_mean', 
     'NUMBER_HEATED_ROOMS_e_minus_mean',
     'Latitude_m', 
     'Longitude_m', 
     'FLOOR_LEVEL_e_minus_mean'
    ], 
    shap_values_misvalued_low, 
    X_misvalued_low_t,  
    display_features = X_misvalued_low_t,  
    rotation=33,
    figsize=(47.5, 40), 
    alpha=1, 
    x_jitter=0.5,
    nrows=5,
    ncols=2,
    xlabelfontsize=24,
    xtickfontsize=20,
    xticklabelrotation=0,
    ylabelfontsize=24,
    ytickfontsize=21,
    cbarlabelfontsize=22,
    cbartickfontsize=20,
    cbartickrotation=0
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 9: Interesting Feature Interactions
# MAGIC 
# MAGIC See the [Tree SHAP paper](https://arxiv.org/pdf/1802.03888.pdf) for more details, but briefly, [SHAP interaction values](https://christophm.github.io/interpretable-ml-book/shap.html#shap-interaction-values) are a generalization of SHAP values to higher order interactions. Fast exact computation of pairwise interactions are implemented in the latest version of XGBoost with the pred_interactions flag. With this flag XGBoost returns a matrix for every prediction, where the main effects are on the diagonal and the interaction effects are off-diagonal. The main effects are similar to the SHAP values you would get for a linear model, and the interaction effects captures all the higher-order interactions are divide them up among the pairwise interaction terms. Note that the sum of the entire interaction matrix is the difference between the model's current output and expected output, and so the interaction effects on the off-diagonal are split in half (since there are two of each). When plotting interaction effects the SHAP package automatically multiplies the off-diagonal values by two to get the full interaction effect.
# MAGIC 
# MAGIC A summary plot of a SHAP interaction value matrix plots a matrix of summary plots with the main effects on the diagonal and the interaction effects off the diagonal.

# COMMAND ----------

shap_interaction_values = explainer.shap_interaction_values(Xs)

# COMMAND ----------


shap.summary_plot(shap_interaction_values, Xs, 
                  max_display=15
                 )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 10: Interesting Interactions Cont'd
# MAGIC 
# MAGIC Now we plot the interaction effects. These effects capture all of the vertical dispersion that was present in the original SHAP plot but is missing from the `main effects plot` (ie the dependence_plot with self-pair (`featureA`,`featureA`)). The plots below show that:.
# MAGIC 
# MAGIC - Lower `A` interacts with `B` such that ...
# MAGIC - Having a `C` is associated with lower probability of `D` when `E` is greater than approx. `X`

# COMMAND ----------

[ 
     'TOTAL_FLOOR_AREA_e',
     'NUMBER_HEATED_ROOMS_e', 
     'FLOOR_LEVEL_e', 
     'Price_p__median', 
     'Price_p__mean', 
     'TOTAL_FLOOR_AREA_e_minus_mean', 
     'NUMBER_HEATED_ROOMS_e_minus_mean',
     'Latitude_m', 
     'Longitude_m', 
     'FLOOR_LEVEL_e_minus_mean'
    ]

plot_dependence = partial(
    _dependence_plot, 
    feature_names=features,
    category_map={},
)

plot_dependence(
    [('TOTAL_FLOOR_AREA_e', 'Price_p__median'),
    ('NUMBER_HEATED_ROOMS_e', 'Price_p__median'), 
    ('FLOOR_LEVEL_e', 'Price_p__median'), 
    ('density_count', 'Price_p__median'), 
    ], 
    shap_interaction_values, 
    Xs, 
    figsize=(30,16.5), 
    rotation=15, 
    ncols=2, 
    nrows=2,
    display_features=Xs,
    xtickfontsize=20,
    xlabelfontsize=20,
    ylabelfontsize=20,
    ytickfontsize=17,
    cbarlabelfontsize=20,
    cbartickfontsize=18,
)

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
# Prepare the dataset
# y_true, y_predicted
test = pd.DataFrame({"Predicted":y_predicted,"Actual":y_true})
test = test.reset_index()
test = test.drop(["index"], axis=1)
 
# plot graphs
fig= plt.figure(figsize=(16,8))
plt.plot(test[:50])
plt.legend(["Actual", "Predicted"])
sns.jointplot(x="Actual", y="Predicted", data=test, kind="reg");