from scipy.io import whosmat, loadmat
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
from matplotlib.gridspec import GridSpec
import seaborn as sns 
import openai 
import re, os, sys, time, math 
from scipy.stats import pearsonr, spearmanr
from transcript_map import transcript_config_all, post_question_dict_all
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency



def _get_value_from_str(pupil_string):
    pupil_item = 0
    pupil_splits = pupil_string.split('-')
    len_item = 0

    for pupil_each in pupil_splits:
        if pupil_each == 'nan': continue
        pupil_item += float(pupil_each)
        len_item += 1
    return pupil_item/len_item




def generate_pearson_result_question(predict_result_path):
    predict_result_table = pd.read_csv(predict_result_path)
    data_prev = predict_result_table[predict_result_table['predict_type']=='type_2_ind_course_confidence']
    data_prev_arr = np.array(data_prev)
    past_score_avg = []
    pre_test_list = []
    isc_list = []
    for i in range(len(data_prev_arr)):
        data_prev_arr_item = data_prev_arr[i]
        past_score_item = data_prev_arr_item[-3]
        pre_test_list.append(data_prev_arr_item[-2])
        isc_list.append(data_prev_arr_item[-1])
        past_score_splits = past_score_item.split('-')
        score_avg = 0
        for score in past_score_splits:
            score_avg += float(score)
        score_avg = score_avg / len(past_score_splits)

        past_score_avg.append(score_avg)

    # type_name_dict = {'label': 'Human', 'type_1_total_only_course': 'Type 1a', 'type_2_total_course_confidence': 'Type 1b', 'type_3_total_course_confidence_pretest': 'Type 1c', 'type_1_only_course': 'Type 2a', 'type_2_course_confidence': 'Type 2b', 'type_3_course_confidence_pretest': 'Type 2c', 'type_2_ind_course_confidence': 'Type 3a', 'type_3_ind_course_confidence_pretest': 'Type 3b', 'type_4_ind_course_confidence_pretest_isc': 'Type 3c'}
    type_name_dict = {'label': 'Hu', 'type_1_total_only_course': '1a', 'type_2_total_course_confidence': '1b', 'type_3_total_course_confidence_pretest': '1c', 'type_1_only_course': '2a', 'type_2_course_confidence': '2b', 'type_3_course_confidence_pretest': '2c', 'type_2_ind_course_confidence': '3a', 'type_3_ind_course_confidence_pretest': '3b', 'type_4_ind_course_confidence_pretest_isc': '3c'}
    
    # predict_result_table = predict_result_table[(predict_result_table['predict_type']!='type_1_total_only_course')&(predict_result_table['predict_type']!='type_2_total_course_confidence')&(predict_result_table['predict_type']!='type_3_total_course_confidence_pretest')]
    # run_id_list = list(set(predict_result_table['predict_type']))
    # run_id_list.sort()
    run_id_list = ['label','type_1_total_only_course','type_2_total_course_confidence','type_3_total_course_confidence_pretest','type_1_only_course', 'type_2_course_confidence', 'type_3_course_confidence_pretest', 'type_2_ind_course_confidence', 'type_3_ind_course_confidence_pretest', 'type_4_ind_course_confidence_pretest_isc']
    data = {}
    for run_id in run_id_list:
        table_item = predict_result_table[predict_result_table['predict_type']==run_id]
        # data[type_name_dict[str(run_id)]] = list(table_item['avg_accuracy_predict'].values)
        data[run_id] = list(table_item['avg_accuracy_predict'].values)
    
    data['U'] = past_score_avg
    data['Pre'] = pre_test_list
    data['ISC'] = isc_list
    df = pd.DataFrame(data)

    return df





def generate_pearson_result_node(result_path):
    corr_df = []
    predict_result_table = pd.read_csv(result_path)
    course_list = list(set(predict_result_table['course_name']))
    for c,course_name in enumerate(course_list):
        predict_result_table_course = predict_result_table[predict_result_table['course_name']==course_name]
        transcript_id_list= list(set(predict_result_table_course['transcript_id']))
        transcript_id_list.sort()
        for it,transcript_id in enumerate(transcript_id_list):
            predict_result_table_item = predict_result_table_course[predict_result_table_course['transcript_id']==transcript_id]
            run_id_list = ['label','all','each','past']

            data = {}
    
            for run_id in run_id_list:
                table_item = predict_result_table_item[predict_result_table_item['predict_type']==run_id]
                data[run_id] = list(table_item['pupil_size_predict'].values)

            df = pd.DataFrame(data)

            for predict_type in ['all','each','past']:
                pearson_r, _ = pearsonr(df['label'], df[predict_type])
                corr_df.append([transcript_id,predict_type,pearson_r,course_name])

    corr_df = pd.DataFrame(np.array(corr_df),columns=['transcript_id','predict_type','pearson_r','course_name'])
    
    return corr_df 

def generate_pearson_trend(predict_result_path,course_name):
    predict_result_table = pd.read_csv(predict_result_path)
    predict_result_table = predict_result_table[predict_result_table['past_pupil_list']!='-']
    predict_result_table['past_u'] = predict_result_table['past_pupil_list'].apply(_get_value_from_str)
    predict_result_table = predict_result_table[predict_result_table['course_name']==course_name]
    transcript_id_list = list(set(predict_result_table['transcript_id']))

    pearson_trend_list = []
    corr_past_list = []
    
    for it,transcript_id in enumerate(transcript_id_list):
        predict_result_table_item = predict_result_table[predict_result_table['transcript_id']==transcript_id]
        run_id_list = list(set(predict_result_table_item['predict_type']))
        run_id_list.sort()

        data = {}
    
        for run_id in run_id_list:
            table_item = predict_result_table_item[predict_result_table_item['predict_type']==run_id]
            data[str(run_id)] = list(table_item['pupil_size_predict'].values)
        
        data['past_u'] = list(table_item['past_u'].values)

        df = pd.DataFrame(data)

        column_list = ['all','each','past','label','past_u']

        for col1 in column_list:
            for col2 in column_list:
                if col1 == 'label' and col2 not in ['label','past_u']:
                    pearson_r_temp, _ = pearsonr(df[col1], df[col2])
                    pearson_trend_list.append([transcript_id,col2,round(pearson_r_temp,2)])
                if col1 == 'past_u' and col2 not in ['past_u']:
                    pearson_u_temp, _ = pearsonr(df[col1], df[col2])
                    corr_past_list.append([transcript_id,col2,round(pearson_u_temp,2)])

    pearson_trend_data = pd.DataFrame(np.array(pearson_trend_list),columns=['transcript_id','predict_type','pearson_r'])    
    pearson_trend_data['pearson_r'] = pearson_trend_data['pearson_r'].astype(float)

    corr_past = pd.DataFrame(np.array(corr_past_list),columns=['transcript_id','predict_type','pearson_r'])    
    corr_past['pearson_r'] = corr_past['pearson_r'].astype(float)

    return pearson_trend_data,corr_past


def func_discretize_isc(raw_value):
    if raw_value < 0.5*0.4:
        return 0
    else:
        return 1

def func_discretize_pre_test(raw_value):
    if raw_value < 0.5:
        return 0
    else:
        return 1


def func_discretize_understand(raw_value):
    if raw_value < 0.5*0.67:
        return 0
    else:
        return 1

def func_discretize_node(raw_value,delta):
    if raw_value < 0.5*delta:
        return 0
    else:
        return 1



def func_discretize_pre_test_draw(raw_value):
    if raw_value['predict_type'] == 'type_2_course_confidence':
        if raw_value['avg_accuracy_predict'] < 0.5:
            return 0
        else:
            return 1
    else:
        if raw_value['avg_accuracy_predict'] < 0.5:
            return 2
        else:
            return 3



def get_pearson_table_node(node_result_file,course_name):
    node_table = pd.read_csv(node_result_file)
    if course_name == 'all':
        node_table = node_table[node_table['past_pupil_list']!='-']
    else:
        node_table = node_table[(node_table['past_pupil_list']!='-')&(node_table['course_name']==course_name)]
    # 
    student_list = list(set(node_table['student_id']))
    # course_list = list(set(node_table['course_name']))
    course_list = [course_name] if course_name != 'all' else list(set(node_table['course_name']))
    
    predict_type_list = ['each','all','past','label']
    new_table = []
    for s,student_id in enumerate(student_list):
        for c,course_name in enumerate(course_list):
            node_part = node_table[(node_table['student_id']==student_id)&(node_table['course_name']==course_name)]
            transcript_list = list(set(node_part['transcript_id']))
            for t,transcript_id in enumerate(transcript_list):
                predict_dict = {}
                for p,predict_type in enumerate(predict_type_list):
                    # node_item = node_table[(node_table['student_id']==student_id)&(node_table['course_name']==course_name)&(node_table['transcript_id']==transcript_id)&(node_table['predict_type']==predict_type)]
                    node_item = node_part[(node_part['transcript_id']==transcript_id)&(node_part['predict_type']==predict_type)]
                    if len(node_item) == 0: break
                    if len(node_item) != 1: assert 1==0
                    predict_dict[predict_type] = node_item['pupil_size_predict'].values[0]
                    u_level = _get_value_from_str(node_item['past_pupil_list'].values[0])
                if len(predict_dict.keys())==0: continue
                new_table.append([predict_dict['each'],predict_dict['all'],predict_dict['past'],predict_dict['label'],u_level])
    return pd.DataFrame(np.array(new_table),columns=['each','all','past','label','understanding_level'])

def check_trend_with_correlation(pupil_string):
    pupil_splits = pupil_string.split('-')
    pupil_list = []

    for pupil_each in pupil_splits:
        if pupil_each == 'nan': continue
        pupil_list.append(float(pupil_each))

    arr = np.array(pupil_list)

    if len(arr) < 2:
        return None

    index = np.arange(len(arr))
    correlation_coefficient, _ = pearsonr(index, arr)

    if correlation_coefficient > 0:
        return 'Increase \u2191'
    else:
        return 'Decrease \u2193'

def f4_main(node_result_file,fontsize):
    sns.set(style="white")
    def _get_dot_size(pearson_value):
        return (float(pearson_value)+1)/2 * 20


    fig,axes = plt.subplots(2,4,figsize=(18,8))
    title_dict = {0:'Birth: Correlation with Labels',1:'Star: Correlation with Labels',2:'Birth: Correlation with Past $U_{level}$',3:'Star: Correlation with Past $U_{level}$'}
    pearson_trend_data_birth,corr_past_birth = generate_pearson_trend(node_result_file,'birth')
    pearson_trend_data_star,corr_past_star = generate_pearson_trend(node_result_file,'star')
    pearson_trend_data = [pearson_trend_data_birth.copy(),pearson_trend_data_star.copy(),corr_past_birth.copy(),corr_past_star.copy()]
    for i,data_item in enumerate(pearson_trend_data):
        data_item['transcript_id'] = data_item['transcript_id'].astype(float)
        data_item['Model'] = data_item['predict_type'].replace({'each': '$T_{ind}$', 'all': '$T_{whole}$', 'past': '$T_{past}$', 'label': '$Label$'})
        # sns.lineplot(data=data_item,x='transcript_id',y='pearson_r',hue='Simulation',marker='o',ax=axes[0][i],markersize=16,linestyle='dashed',alpha=0.7)
        sns.scatterplot(data=data_item,x='transcript_id',y='pearson_r',hue='Model',marker='o',ax=axes[0][i],alpha=0.7)
        for category in data_item['Model'].unique():
            subset = data_item[data_item['Model'] == category]
            sns.regplot(x='transcript_id',y='pearson_r', data=subset, scatter=False, ax=axes[0][i], order=2 ) # lowess=True
        axes[0][i].set_title(title_dict[i], fontsize=fontsize)
        axes[0][i].spines['top'].set_visible(False)
        axes[0][i].spines['right'].set_visible(False)
        axes[0][i].set_xlabel('Slide ID', fontsize=fontsize)
        axes[0][i].set_ylabel('Pearson $r$', fontsize=fontsize)
        axes[0][i].tick_params(axis='x', labelsize=fontsize) 
        axes[0][i].tick_params(axis='y', labelsize=fontsize) 

    node_result_raw = pd.read_csv(node_result_file)
    node_corr = generate_pearson_result_node(node_result_file)
 
    pearson_table_sub_star = get_pearson_table_node(node_result_file,'all')
    type_name_dict = {'each': '$T_{ind}$','all':'$T_{whole}$','past':'$T_{past}$','label':'$Label$','understanding_level':'$U_{level}$'}
    for tp in type_name_dict:
        pearson_table_sub_star[type_name_dict[tp]] = pearson_table_sub_star[tp]
    pearson_table_sub_star = pearson_table_sub_star[list(type_name_dict.values())]
    correlation_matrix = pearson_table_sub_star.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool),k=1)
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='Oranges', fmt=".2f", mask=mask, annot_kws={"fontsize":10},ax=axes[1][0])
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    node_result = node_result_raw.copy()
    node_result = node_result[node_result['past_pupil_list']!='-']
    node_result['understanding_level_raw'] = node_result['past_pupil_list'].apply(_get_value_from_str)
    node_result['Past $U_{trend}$'] = node_result['past_pupil_list'].apply(check_trend_with_correlation)
    understanding_level_arr = np.array(node_result['understanding_level_raw'])
    u_max, u_min = np.max(understanding_level_arr), np.min(understanding_level_arr)
    node_result['Understanding Level'] = node_result['understanding_level_raw'].apply(func_discretize_node,args=(u_max - u_min,))
    node_result['Past $U_{level}$'] = node_result['Understanding Level'].replace({0:'Low',1:'High'})
    node_result['predict_type'] = node_result['predict_type'].replace({'each': '$T_{ind}$', 'all': '$T_{whole}$', 'past': '$T_{past}$', 'label': '$Label$'})
    sns.stripplot(data=node_result, x="predict_type", y="pupil_size_predict", hue="Past $U_{level}$",dodge=True, alpha=.2, legend=False,ax=axes[1][1])
    sns.pointplot(data=node_result, x="predict_type", y="pupil_size_predict", hue="Past $U_{level}$",dodge=0.4, linestyle="none", errorbar=None,marker="_", markersize=15, markeredgewidth=3,ax=axes[1][1])
    axes[1][1].spines['top'].set_visible(False)
    axes[1][1].spines['right'].set_visible(False)
    axes[1][1].set_xlabel('Model', fontsize=fontsize)
    axes[1][1].set_ylabel('$U_{level}$ Prediction', fontsize=fontsize)
    axes[1][1].tick_params(axis='x', labelsize=fontsize) 
    axes[1][1].tick_params(axis='y', labelsize=fontsize) 

    sns.stripplot(data=node_result, x="predict_type", y="pupil_size_predict", hue="Past $U_{trend}$",dodge=True, alpha=.2, legend=False,ax=axes[1][2])
    sns.pointplot(data=node_result, x="predict_type", y="pupil_size_predict", hue="Past $U_{trend}$",dodge=0.4, linestyle="none", errorbar=None,marker="_", markersize=15, markeredgewidth=3,ax=axes[1][2])
    axes[1][2].spines['top'].set_visible(False)
    axes[1][2].spines['right'].set_visible(False)
    axes[1][2].set_xlabel('Model', fontsize=fontsize)
    axes[1][2].set_ylabel('$U_{level}$ Prediction', fontsize=fontsize)
    axes[1][2].tick_params(axis='x', labelsize=fontsize) 
    axes[1][2].tick_params(axis='y', labelsize=fontsize) 


    node_result_raw['Model'] = node_result_raw['predict_type'].replace({'each': '$T_{ind}$', 'all': '$T_{whole}$', 'past': '$T_{past}$', 'label': '$Label$'})
    sns.ecdfplot(node_result_raw,x='pupil_size_predict',hue='Model',ax=axes[1][3])
    axes[1][3].spines['top'].set_visible(False)
    axes[1][3].spines['right'].set_visible(False)
    axes[1][3].set_xlabel('$U_{level}$', fontsize=fontsize)
    axes[1][3].set_ylabel('Cumulative Probability', fontsize=fontsize)
    axes[1][3].tick_params(axis='x', labelsize=fontsize) 
    axes[1][3].tick_params(axis='y', labelsize=fontsize) 


    letter_ids = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']]

    for i, letter_list in enumerate(letter_ids):
        for j, letter in enumerate(letter_list):
            axes[i, j].text(-0.25, 1.1, letter, fontsize=fontsize+15, transform=axes[i, j].transAxes)


    fig.tight_layout()
    fig.savefig('understand.pdf')
    plt.show()


def f6_main(question_item_result_path,pupil_result_path,question_avg_result_path,course_name):
    fig = plt.figure(figsize=(12, 11))
    
    gs = GridSpec(5, 12, figure=fig, hspace=0.5, wspace=1)
    all_axes = []

    for i in range(11):
        row, col = divmod(i, 6)  # Calculate row and column index
        if i in [0,1,2]:
            new_ax = fig.add_subplot(gs[i:i+1, 0:6])
        # elif i in [2]:
        #     new_ax = fig.add_subplot(gs[2:4, 0:6])
        elif i in [3,4]:
            new_ax = fig.add_subplot(gs[0:3, 6+(i-3)*3:6+(i-3)*3+3])
        else:
            new_ax = fig.add_subplot(gs[3:, (i-5)*2:(i-5)*2+2])
        all_axes.append(new_ax)
        
    def _get_dot_size(pearson_value):
        return (float(pearson_value)+1)/2 * 20

    sns.set(style="white")
    fontsize = 16
    ax31, ax41, ax51 = all_axes[0], all_axes[1], all_axes[2]
    
    question_result = pd.read_csv(question_avg_result_path)
    question_result_df = generate_pearson_result_question(question_avg_result_path)
    
    pearson_result_question_list = []
    run_id_list = ['type_1_total_only_course','type_2_total_course_confidence','type_3_total_course_confidence_pretest','type_1_only_course', 'type_2_course_confidence', 'type_3_course_confidence_pretest', 'type_2_ind_course_confidence', 'type_3_ind_course_confidence_pretest', 'type_4_ind_course_confidence_pretest_isc']

    pearson_result_question_list.append(pearsonr(question_result_df['type_2_total_course_confidence'], question_result_df['label'])[0])
    pearson_result_question_list.append(pearsonr(question_result_df['type_2_course_confidence'], question_result_df['label'])[0])
    pearson_result_question_list.append(pearsonr(question_result_df['type_2_ind_course_confidence'], question_result_df['label'])[0])
    pearson_result_question_list.append(pearsonr(question_result_df['type_3_total_course_confidence_pretest'], question_result_df['label'])[0])
    pearson_result_question_list.append(pearsonr(question_result_df['type_3_course_confidence_pretest'], question_result_df['label'])[0])
    pearson_result_question_list.append(pearsonr(question_result_df['type_3_ind_course_confidence_pretest'], question_result_df['label'])[0])
    pearson_result_question_list.append(pearsonr(question_result_df['type_2_ind_course_confidence'], question_result_df['label'])[0])
    pearson_result_question_list.append(pearsonr(question_result_df['type_3_ind_course_confidence_pretest'], question_result_df['label'])[0])
    pearson_result_question_list.append(pearsonr(question_result_df['type_4_ind_course_confidence_pretest_isc'], question_result_df['label'])[0])

    pearson_result_question_table = pd.DataFrame({
                    'pearson': pearson_result_question_list,
                    # 'type': ['Ta','Tb','Tc','Ta','Tb','Tc','Ta','Tb','Tc'],
                    'type': ['$T_{whole}$','$T_{item}$','$T_{context}$','$T_{whole}$','$T_{item}$','$T_{context}$','$T_{whole}$','$T_{item}$','$T_{context}$'],
                     'exp': ['Contextual','Contextual','Contextual','+ Pre Test','+ Pre Test','+ Pre Test','+ Engagement','+ Engagement','+ Engagement']})
                    # 'type': ['T0','Ta','Ta','T0','Tb','Tb','Tc','Tc','Ta'],
                    #  'exp': ['Exp 0','Exp 5','Exp 6','Exp 0','Exp 5','Exp 6','Exp 5','Exp 6','Exp 7']})

    color_predict_type_dict_ax31 = {"$Label$": "#808080", "$T_{whole}$": "#2ca02c", "$T_{item}$": "#1f77b4", "$T_{context}$": '#ff7f0e'}  # Your color dictionary
    custom_palette_ax31 = {type_str: color_predict_type_dict_ax31[type_str] for type_str in set(pearson_result_question_table['type'])}

    sns.barplot(pearson_result_question_table,x='exp',y='pearson',hue='type',alpha=0.95,ax=ax31,palette=custom_palette_ax31,legend=False)
    for p in ax31.patches:
        ax31.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 2), textcoords='offset points', fontsize=8)

    ax31.spines['top'].set_visible(False)
    ax31.spines['right'].set_visible(False)
    ax31.text(-0.7, 0.7, 'a', fontsize=20, color='black')
    ax31.set_title('Correlation between predictions and labels in post-test score',fontsize=10)
    ax31.set_xlabel('')
    ax31.set_ylabel('Pearson $r$')
    # ax31.legend(bbox_to_anchor=(0.5, 1.4), loc='upper center', ncol=3, fontsize=10)

    pearson_corr_question_list = []
    run_id_list = ['type_1_total_only_course','type_2_total_course_confidence','type_3_total_course_confidence_pretest','type_1_only_course', 'type_2_course_confidence', 'type_3_course_confidence_pretest', 'type_2_ind_course_confidence', 'type_3_ind_course_confidence_pretest', 'type_4_ind_course_confidence_pretest_isc']
    
    pearson_corr_question_list.append(pearsonr(question_result_df['label'], question_result_df['U'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_2_total_course_confidence'], question_result_df['U'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_2_course_confidence'], question_result_df['U'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_2_ind_course_confidence'], question_result_df['U'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['label'], question_result_df['Pre'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_3_total_course_confidence_pretest'], question_result_df['Pre'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_3_course_confidence_pretest'], question_result_df['Pre'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_3_ind_course_confidence_pretest'], question_result_df['Pre'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['label'], question_result_df['ISC'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_2_ind_course_confidence'], question_result_df['ISC'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_3_ind_course_confidence_pretest'], question_result_df['ISC'])[0])
    pearson_corr_question_list.append(pearsonr(question_result_df['type_4_ind_course_confidence_pretest_isc'], question_result_df['ISC'])[0])



    pearson_corr_question_table = pd.DataFrame({
                    'pearson': pearson_corr_question_list,
                    # 'type': ['label','Ta','Tb','Tc','label','Ta','Tb','Tc','label','Ta','Tb','Tc'],
                    'type': ['$Label$','$T_{whole}$','$T_{item}$','$T_{context}$','$Label$','$T_{whole}$','$T_{item}$','$T_{context}$','$Label$','$T_{whole}$','$T_{item}$','$T_{context}$'],
                    # 'type': ['$Label$','$T_{whole}$','$T_{item}$','$T_{context}$','$Label$','$T_{whole-pre}$','$T_{item-pre}$','$T_{context-pre}$','$Label$','$T_{whole-pre-engage}$','$T_{item-pre-engage}$','$T_{context-pre-engage}$'],
                     'exp': ['$r_{PostTest-Understanding}$','$r_{PostTest-Understanding}$','$r_{PostTest-Understanding}$','$r_{PostTest-Understanding}$','$r_{PostTest-Pre Test}$','$r_{PostTest-Pre Test}$','$r_{PostTest-Pre Test}$','$r_{PostTest-Pre Test}$','$r_{PostTest-Engagement}$','$r_{PostTest-Engagement}$','$r_{PostTest-Engagement}$','$r_{PostTest-Engagement}$']})
    
    color_predict_type_dict_ax41 = {"$Label$": "#808080", "$T_{whole}$": "#2ca02c", "$T_{item}$": "#1f77b4", "$T_{context}$": '#ff7f0e'}  # Your color dictionary
    custom_palette_ax41 = {type_str: color_predict_type_dict_ax41[type_str] for type_str in set(pearson_corr_question_table['type'])}
    
    sns.barplot(pearson_corr_question_table,x='exp',y='pearson',hue='type',alpha=0.95,ax=ax41,palette=custom_palette_ax41,legend=True)
    for p in ax41.patches:
        ax41.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 2), textcoords='offset points', fontsize=8)

    ax41.spines['top'].set_visible(False)
    ax41.spines['right'].set_visible(False)
    ax41.text(-0.7, 1, 'b', fontsize=20, color='black')
    ax41.set_ylabel('Pearson $r$')
    ax41.set_xlabel('')
    ax41.set_title('Correlation between post-test score (prediction/label) and past behaviors',fontsize=10)
    ax41.legend(bbox_to_anchor=(0.5, 3), loc='upper center', ncol=4, fontsize=10)

    question_result['understanding_level_raw'] = question_result['past_pupil'].apply(_get_value_from_str)
    u_arr = np.array(question_result['understanding_level_raw'])
    u_max,u_min = np.max(u_arr),np.min(u_arr)
    pre_test_arr = np.array(question_result['pre_test'])
    pre_test_max,pre_test_min = np.max(pre_test_arr),np.min(pre_test_arr)
    isc_arr = np.array(question_result['isc'])
    isc_max,isc_min = np.max(isc_arr),np.min(isc_arr)

    predict_type_arr = np.array(question_result['predict_type'])
    accuracy_predict_arr = np.array(question_result['avg_accuracy_predict'])
    factor_discrete_list = []
    question_result_aggregate = [] 
    shft = 0
    predict_type_name_dict_1 = {'label':'human','type_4_ind_course_confidence_pretest_isc':'t3','type_2_ind_course_confidence':'t1','type_3_ind_course_confidence_pretest':'t2'}
    for i in range(len(pre_test_arr)):
        if predict_type_arr[i] in ['label','type_4_ind_course_confidence_pretest_isc','type_2_ind_course_confidence','type_3_ind_course_confidence_pretest']:
            if isc_arr[i] < 0.5*(isc_max-isc_min):
                question_result_aggregate.append([4+shft,accuracy_predict_arr[i],predict_type_name_dict_1[predict_type_arr[i]]+'_e7'])
            else:
                question_result_aggregate.append([5+shft,accuracy_predict_arr[i],predict_type_name_dict_1[predict_type_arr[i]]+'_e7'])

    predict_type_name_dict_2 = {'label':'human','type_3_ind_course_confidence_pretest':'t3','type_3_course_confidence_pretest':'t2','type_3_total_course_confidence_pretest':'t1'}
    for i in range(len(pre_test_arr)):
        if predict_type_arr[i] in ['label','type_3_course_confidence_pretest','type_3_total_course_confidence_pretest','type_3_ind_course_confidence_pretest']:
            if pre_test_arr[i] < 0.5*(pre_test_max-pre_test_min):
                question_result_aggregate.append([2+shft,accuracy_predict_arr[i],predict_type_name_dict_2[predict_type_arr[i]]+'_e6'])
            else:
                question_result_aggregate.append([3+shft,accuracy_predict_arr[i],predict_type_name_dict_2[predict_type_arr[i]]+'_e6'])
        
    predict_type_name_dict_3 = {'label':'human','type_2_course_confidence':'t2','type_2_total_course_confidence':'t1','type_2_ind_course_confidence':'t3'}
    for i in range(len(pre_test_arr)):
        if predict_type_arr[i] in ['label','type_2_course_confidence','type_2_total_course_confidence','type_2_ind_course_confidence']:
            if u_arr[i] < 0.5*(u_max-u_min):
                question_result_aggregate.append([0+shft,accuracy_predict_arr[i],predict_type_name_dict_3[predict_type_arr[i]]+'_e5'])
            else:
                question_result_aggregate.append([1+shft,accuracy_predict_arr[i],predict_type_name_dict_3[predict_type_arr[i]]+'_e5'])
        

    question_result_aggregate = pd.DataFrame(np.array(question_result_aggregate),columns=["factor_discrete","avg_accuracy_predict","predict_type"])
    question_result_aggregate['factor_discrete'] = question_result_aggregate['factor_discrete'].astype(float)
    question_result_aggregate['avg_accuracy_predict'] = question_result_aggregate['avg_accuracy_predict'].astype(float)

    color_predict_type_dict = {"human_e5": "#808080", "t1_e5": "#2ca02c", "t2_e5": "#1f77b4", "t3_e5": '#ff7f0e', "human_e6": "#808080", "t1_e6": "#2ca02c", "t2_e6": "#1f77b4", "t3_e6": '#ff7f0e', "human_e7": "#808080", "t1_e7": "#2ca02c", "t2_e7": "#1f77b4", "t3_e7": '#ff7f0e'}  # Your color dictionary
    custom_palette = {predict_type: color_predict_type_dict[predict_type] for predict_type in set(question_result_aggregate['predict_type'])}
    
    sns.pointplot(data=question_result_aggregate, x="factor_discrete", y="avg_accuracy_predict", hue="predict_type", dodge=True, ax=ax51, palette=custom_palette, legend=False)
    # ax51.legend([])
    # ax51.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1, fontsize=6)
    # ax51.legend(loc='upper center', bbox_to_anchor=(0.5, 2),
        #    fancybox=True, shadow=True, ncol=6)
    ax51.spines['top'].set_visible(False)
    ax51.spines['right'].set_visible(False)
    ax51.text(-0.85, 1, 'c', fontsize=20, color='black')
    ax51_x_axis = np.arange(0, 6, 1)
    ax51.set_xticks(ax51_x_axis)
    tick_label_dict = {0:'$Low_{Understanding}$',1:'$High_{Understanding}$',2:'$Low_{PreTest}$',3:'$Low_{PreTest}$',4:'$Low_{Engagement}$',5:'$Low_{Engagement}$'}
    ax51.set_xticklabels([tick_label_dict.get(x_i, str(x_i)) for x_i in ax51_x_axis],fontsize=7)
    ax51.set_xlabel('')
    ax51.set_ylabel('Post-Test Score')
    ax51.set_title('Post-test score (prediction/label) in various past behaviors levels',fontsize=10)

    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    ax1 = all_axes[3]
    ax2 = all_axes[4]
    ax3 = all_axes[5]
    ax4 = all_axes[6]
    ax5 = all_axes[7]
    ax6 = all_axes[8]
    ax7 = all_axes[9]
    ax8 = all_axes[10]

    pupil_data = pd.read_csv(pupil_result_path)
    pupil_data = pupil_data[pupil_data['course_name']==course_name]
    question_item_data = pd.read_csv(question_item_result_path)
    
    question_item_data = question_item_data[(question_item_data['course_name']==course_name)]
    student_id_list = list(set(question_item_data['student_id']))
    student_id_list.sort()
    question_id_list = list(set(question_item_data['question_id']))
    question_id_list.sort()
    pupil_matrix = []
    
    predict_matrix_dict = {}
    predict_type_list = list(set(question_item_data['predict_type']))
    predict_type_list.sort()
    predict_type_list = ['type_1_only_course', 'type_2_course_confidence', 'type_3_course_confidence_pretest', 'type_2_ind_course_confidence', 'type_3_ind_course_confidence_pretest', 'type_4_ind_course_confidence_pretest_isc']

    
    for p,predict_type in enumerate(predict_type_list):
        predict_matrix = []
        for i,student_id in enumerate(student_id_list):
            predict_matrix_sub = []
            for j,question_id in enumerate(question_id_list):
                question_item_data_item = question_item_data[(question_item_data['student_id']==student_id)&(question_item_data['question_id']==question_id)&(question_item_data['predict_type']==predict_type)]
                predict_matrix_sub.append(question_item_data_item['accuracy_per_predict'].values[0])
            predict_matrix.append(predict_matrix_sub)
        predict_matrix_dict[predict_type] = predict_matrix

    sub_matrix_avg = []
    for i,student_id in enumerate(student_id_list):
        sub_matrix = []
        
        for j,question_id in enumerate(question_id_list):
            pupil_avg = 0
            slide_id_list = post_question_dict_all[course_name][question_id]['slide_id_list']
            for k,slide_id in enumerate(slide_id_list):
                pupil_item = pupil_data[(pupil_data['student_id']==student_id)&(pupil_data['transcript_id']==slide_id)]
                pupil_avg += (1-pupil_item['pupil_size_rela'].values[0])
            sub_matrix.append(pupil_avg/len(slide_id_list))
            
        pupil_matrix.append(sub_matrix)
        sub_matrix_avg.append(np.mean(sub_matrix))

    
    sns.heatmap(pupil_matrix, annot=False, cmap='Reds', fmt=".2f", linewidths=1, ax=ax1, cbar=False, xticklabels=False, yticklabels=False)
    ax1.set_title('Understanding Level',fontsize=12)
    ax1.set_xlabel('Question ID: 1 $\\rightarrow$ 12',fontsize=12)
    ax1.set_ylabel('Student ID: 27 $\\rightarrow$ 1',fontsize=12)
    ax1.text(-1.5, -0.5, 'd', fontsize=20, color='black')

    axes = {0:ax3,1:ax4,2:ax5,3:ax6,4:ax7,5:ax8}

    num_icon_dict = {0:'f',1:'g',2:'h',3:'i',4:'j',5:'k'}
    # num_icon_dict = {0:'(c)',1:'(d)',2:'(e)',3:'(f)',4:'(g)',5:'(h)'}

    predict_name_dict = {'type_1_only_course': '$T_{only-course}$', 'type_2_course_confidence': '$T_{item}$', 'type_3_course_confidence_pretest': '$T_{item-pre}$', 'type_2_ind_course_confidence': '$T_{context}$', 'type_3_ind_course_confidence_pretest': '$T_{context-pre}$', 'type_4_ind_course_confidence_pretest_isc': '$T_{context-pre-engage}$'}

    # predict_name_dict = {'type_1_only_course': 'Type 2a', 'type_2_course_confidence': 'Type 2b', 'type_3_course_confidence_pretest': 'Type 2c', 'type_2_ind_course_confidence': 'Type 3a', 'type_3_ind_course_confidence_pretest': 'Type 3b', 'type_4_ind_course_confidence_pretest_isc': 'Type 3c'}
    for p,predict_type in enumerate(predict_type_list):
        sns.heatmap(predict_matrix_dict[predict_type], annot=False, cmap='Reds', cbar=False, linewidths=0, xticklabels=False, yticklabels=False, fmt=".2f", ax=axes[p], cbar_kws={'color': 'lightgray'})
        axes[p].set_title(predict_name_dict[predict_type],fontsize=10)
        # if p in [0,1,2,3,4,5]:
        axes[p].set_xlabel('Q: 1$\\rightarrow$12',fontsize=12)
        # if p in [0]:
        axes[p].set_ylabel('Student ID: 27 $\\rightarrow$ 1',fontsize=12)
        
        axes[p].text(-2, -1, num_icon_dict[p], fontsize=20, color='black')

    question_avg_data = pd.read_csv(question_avg_result_path)
    question_avg_data = question_avg_data[question_avg_data['course_name']==course_name]
    # predict_type_avg_list = list(set(question_avg_data['predict_type']))
    # predict_type_avg_list.sort()
    predict_type_avg_list = ['label','type_1_total_only_course','type_2_total_course_confidence','type_3_total_course_confidence_pretest','type_1_only_course', 'type_2_course_confidence', 'type_3_course_confidence_pretest', 'type_2_ind_course_confidence', 'type_3_ind_course_confidence_pretest', 'type_4_ind_course_confidence_pretest_isc']

    question_avg_label_item = question_avg_data[question_avg_data['predict_type']=='label']
    past_pupil_avg_list = list(question_avg_label_item['past_pupil'].values)
    pre_test_avg_list = list(question_avg_label_item['pre_test'].values)
    isc_avg_list = list(question_avg_label_item['isc'].values)

    total_score_matrix = []
    total_score_matrix.append([_get_value_from_str(past_pupil_avg) for past_pupil_avg in past_pupil_avg_list])
    total_score_matrix.append(pre_test_avg_list)
    total_score_matrix.append(isc_avg_list)

    for predict_type_avg in predict_type_avg_list:
        question_avg_data_item = question_avg_data[question_avg_data['predict_type']==predict_type_avg]
        total_score_matrix.append(list(question_avg_data_item['avg_accuracy_predict'].values))

    total_score_matrix = np.array(total_score_matrix)
    total_score_matrix = total_score_matrix.T
    
    heatmap=sns.heatmap(total_score_matrix, annot=False, cmap='Reds', fmt=".2f", linewidths=1, ax=ax2, cbar=False, xticklabels=['U','P','E','L','1a','1b','1c','2a','2b','2c','3a','3b','3c'], yticklabels=False)
    ax2.set_title('Mean Post-Test Score',fontsize=12)
    ax2.set_ylabel('Student ID: 27 $\\rightarrow$ 1',fontsize=12)

    tick_labels = ['    U','    P','    E','    L','     $T_{wo}$','     $T_{w}$','     $T_{wp}$','      $T_{io}$','      $T_{i}$','      $T_{ip}$','      $T_{c}$','      $T_{cp}$','        $T_{cpe}$']
    # tick_labels = ['     U','     P','     E','     L','     $T_{wo}$','     $T_{w}$','     $T_{wp}$','     $T_{io}$','     $T_{i}$','     $T_{ip}$','     $T_{c}$','     $T_{cp}$','     $T_{cpe}$']
    tick_colors = ['black','black','black','#808080','#2ca02c','#2ca02c','#2ca02c','#1f77b4','#1f77b4','#1f77b4','#ff7f0e','#ff7f0e','#ff7f0e']

    # Set the tick positions and labels for the x-axis
    x_ticks = range(len(tick_labels))

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(tick_labels)

    # Set different colors for tick labels
    for i, label in enumerate(ax2.get_xticklabels()):
        label.set_color(tick_colors[i])

    ax2.text(-1, -0.5, 'e', fontsize=20, color='black')
    heatmap.axes.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # ax2.set_xlabel('Understanding, Pre Test, ISC, Type: 1a $\\rightarrow$ 3c',fontsize=12)

    cax2 = fig.add_axes([0.93, 0.12, 0.01, 0.76])  # Adjust the position and size of the color bar
    cbar2 = plt.colorbar(ax2.get_children()[0], cax=cax2, orientation="vertical")
    cbar2.ax.tick_params(labelsize=16)
    

    plt.savefig('matrix_vis.pdf')

    plt.show()

# main text figure

f4_main('result_node.csv',fontsize=14)
f6_main('result_question_item.csv','datasets/student_result.csv','result_question_avg.csv','birth')

