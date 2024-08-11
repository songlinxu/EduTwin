from scipy.io import whosmat, loadmat
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
from matplotlib.gridspec import GridSpec
import seaborn as sns 
import re, os, sys, time, math 
from scipy.stats import pearsonr, spearmanr
from d3blocks import D3Blocks

def chord_plot(corr_dur_slide_baseline):
    corr_dur_slide_table = pd.read_csv(corr_dur_slide_baseline)
    corr_dur_slide_number_one = corr_dur_slide_table[(corr_dur_slide_table['feature_type']=='number')&(corr_dur_slide_table['model_name']=='rnn')]
    corr_dur_slide_llm_one = corr_dur_slide_table[(corr_dur_slide_table['feature_type']=='llm')&(corr_dur_slide_table['model_name']=='rnn')]

    metric_list = ['gaze entropy','workload','curiosity','valid focus','course follow','question accuracy']

    corr_dur_table_sub_number_one = corr_dur_slide_number_one[['agent ' + metric_name for metric_name in metric_list]]
    corr_dur_table_sub_llm_one = corr_dur_slide_llm_one[['agent ' + metric_name for metric_name in metric_list]]
    corr_dur_table_sub_user = corr_dur_slide_number_one[['user ' + metric_name for metric_name in metric_list]]

    corr_dict_ind = {'Human':corr_dur_table_sub_user, 'Number':corr_dur_table_sub_number_one, 'LLM':corr_dur_table_sub_llm_one}

    for chord_name in ['Human','Number','LLM']:
        chord_raw_data = corr_dict_ind[chord_name]
        chord_df = []
        for cmi,chord_source in enumerate(metric_list):
            chord_target_list = metric_list[cmi+1:].copy()
            for chord_target in chord_target_list:
                if chord_name == 'Human':
                    chord_weight = chord_raw_data['user '+chord_source].corr(chord_raw_data['user '+chord_target])
                else:
                    chord_weight = chord_raw_data['agent '+chord_source].corr(chord_raw_data['agent '+chord_target])
                chord_df.append([chord_source,chord_target,chord_weight])

        chord_df = pd.DataFrame(np.array(chord_df),columns = ['source','target','weight'])
        chord_df['weight'] = chord_df['weight'].astype(float)
        chord_df['weight'] = (chord_df['weight'] - chord_df['weight'].min()) / (chord_df['weight'].max() - chord_df['weight'].min())
        print('chord_df',chord_df)

        d3 = D3Blocks()
        d3.chord(chord_df,fontsize=20,arrowhead=10,filepath='./temp_'+chord_name+'.html')



def visual_pearson_lineplot(corr_dur_baseline_csv):
    corr_dur_slide_table_baseline = pd.read_csv(corr_dur_baseline_csv)

    metric_list = ['gaze entropy','workload','curiosity','valid focus','course follow','question accuracy']
    
    fig03,ax03=plt.subplots(3,5,figsize=(10,6))
    
    feature_type_list = ['number','bert','llm']
    model_name_list = ['decision_tree','linear_regression','random_forest','transformer','rnn']
    df_plot = pd.DataFrame()
    
    color_dict = {'number':'orange','bert':'b','llm':'r'}
    feature_str_dict = {'llm': 'LLM','number':'Number','bert':'BERT'}
    model_str_dict = {'decision_tree':'Decision Tree','linear_regression':'Linear Regression','rnn':'RNN','random_forest':'Random Forest','transformer':'Transformer'}
    j = -1
    for feature_type in feature_type_list:
        for model_name in model_name_list:
            j += 1
            row = int(j/5)
            col = j-int(row*5)
            print(feature_type,model_name,row,col)
            corr_dur_slide_table_baseline_per = corr_dur_slide_table_baseline[(corr_dur_slide_table_baseline['feature_type']==feature_type)&(corr_dur_slide_table_baseline['model_name']==model_name)]

            corr_dur_table_sub_model = corr_dur_slide_table_baseline_per[['agent ' + metric_name for metric_name in metric_list]]
            corr_dur_table_sub_label = corr_dur_slide_table_baseline_per[['user ' + metric_name for metric_name in metric_list]]
        
            corr_dur_table_sub_label = corr_dur_table_sub_label.dropna()
            corr_dur_table_sub_model = corr_dur_table_sub_model.dropna()

            correlation_matrix_model = corr_dur_table_sub_model.corr()
            correlation_matrix_label = corr_dur_table_sub_label.corr()
            
            mask = np.triu(np.ones(correlation_matrix_label.shape), k=1).astype(bool)
            flatten_correlation_matrix_label = correlation_matrix_label.where(mask).stack().reset_index(drop=True)
            flatten_correlation_matrix_model = correlation_matrix_model.where(mask).stack().reset_index(drop=True)

            x_name = feature_type+'/'+model_name+'/pred'
            y_name = feature_type+'/'+model_name+'/label'
            
            df_plot[y_name] = flatten_correlation_matrix_label
            df_plot[x_name] = flatten_correlation_matrix_model

            sns.regplot(x=x_name, y=y_name, data=df_plot, ax=ax03[row][col], line_kws={'color': color_dict[feature_type]}, scatter_kws={'color': color_dict[feature_type]})
            ax03[row][col].spines['right'].set_visible(False)
            ax03[row][col].spines['top'].set_visible(False)
            ax03[row][col].set_xlabel(feature_str_dict[feature_type]+': '+model_str_dict[model_name])
            ax03[row][col].set_ylabel('Real Human')
            try:
                correlation, p_value = pearsonr(df_plot[x_name], df_plot[y_name])
                ax03[row][col].set_title(f"$\\beta$: {correlation:.3f}, $P$: {p_value:.3f}")
            except:
                nan_indices = df_plot[x_name][df_plot[x_name].isna()].index
                df_plot_x = df_plot[x_name].drop(nan_indices)
                df_plot_y = df_plot[y_name].drop(nan_indices)
                correlation, p_value = pearsonr(df_plot_x, df_plot_y)
                ax03[row][col].set_title(f"$\\beta$: {correlation:.3f}, $P$: {p_value:.3f}")
            
    fig03.tight_layout()
    fig03.savefig('pearson.pdf')





# visual_pearson_lineplot('result/corr_dur_slide_baseline_extract.csv')
# chord_plot('result/corr_dur_slide_baseline_extract.csv')


