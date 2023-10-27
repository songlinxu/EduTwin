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

def _get_value_from_str(pupil_string):
    pupil_item = 0
    pupil_splits = pupil_string.split('-')
    for pupil_each in pupil_splits:
        pupil_item += float(pupil_each)
    return pupil_item/len(pupil_splits)





def matrix_question_student_grid(question_item_result_path,pupil_result_path,question_avg_result_path,course_name):
    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    fig = plt.figure(figsize=(12,7))

    gs = GridSpec(2, 7, figure=fig, wspace=0.3, hspace=0.15)
    # gs = GridSpec(2, 7, figure=fig, width_ratios=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    ax1 = fig.add_subplot(gs[:, :2])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax2 = fig.add_subplot(gs[:, 2:4])
    ax3 = fig.add_subplot(gs[:1, 4:5])
    ax4 = fig.add_subplot(gs[:1, 5:6])
    ax5 = fig.add_subplot(gs[:1, 6:7])
    ax6 = fig.add_subplot(gs[1:2, 4:5])
    ax7 = fig.add_subplot(gs[1:2, 5:6])
    ax8 = fig.add_subplot(gs[1:2, 6:7])

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
    ax1.set_title('Understanding Level',fontsize=14)
    ax1.set_xlabel('Question ID: 1 $\\rightarrow$ 12',fontsize=12)
    ax1.set_ylabel('Student ID: 27 $\\rightarrow$ 1',fontsize=12)
    ax1.text(-1.5, -0.5, 'A', fontsize=20, color='black')

    axes = {0:ax3,1:ax4,2:ax5,3:ax6,4:ax7,5:ax8}

    num_icon_dict = {0:'C',1:'D',2:'E',3:'F',4:'G',5:'H'}
    # num_icon_dict = {0:'(c)',1:'(d)',2:'(e)',3:'(f)',4:'(g)',5:'(h)'}

    predict_name_dict = {'type_1_only_course': 'Type 2a', 'type_2_course_confidence': 'Type 2b', 'type_3_course_confidence_pretest': 'Type 2c', 'type_2_ind_course_confidence': 'Type 3a', 'type_3_ind_course_confidence_pretest': 'Type 3b', 'type_4_ind_course_confidence_pretest_isc': 'Type 3c'}
    for p,predict_type in enumerate(predict_type_list):
        # print('predict_type: ',predict_type)
        sns.heatmap(predict_matrix_dict[predict_type], annot=False, cmap='Reds', cbar=False, linewidths=0, xticklabels=False, yticklabels=False, fmt=".2f", ax=axes[p], cbar_kws={'color': 'lightgray'})
        # sns.heatmap(predict_matrix_dict[predict_type], annot=False, cmap='Reds', cbar=False, xticklabels=False, yticklabels=False, linewidths=2, fmt=".2f", ax=axes[p+1])
        axes[p].set_title(predict_name_dict[predict_type],fontsize=14)
        if p not in [0,1,2]:
            axes[p].set_xlabel('Q: 1$\\rightarrow$12',fontsize=12)
        if p in [0,3]:
            axes[p].set_ylabel('Student ID: 27 $\\rightarrow$ 1',fontsize=12)
        
        axes[p].text(-2, -1, num_icon_dict[p], fontsize=20, color='black')

    question_avg_data = pd.read_csv(question_avg_result_path)
    question_avg_data = question_avg_data[question_avg_data['course_name']==course_name]
    # predict_type_avg_list = list(set(question_avg_data['predict_type']))
    # predict_type_avg_list.sort()
    predict_type_avg_list = ['label','type_1_total_only_course','type_2_total_course_confidence','type_3_total_course_confidence_pretest','type_1_only_course', 'type_2_course_confidence', 'type_3_course_confidence_pretest', 'type_2_ind_course_confidence', 'type_3_ind_course_confidence_pretest', 'type_4_ind_course_confidence_pretest_isc']
    print(predict_type_avg_list)

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
        print(predict_type_avg,list(question_avg_data_item['avg_accuracy_predict'].values))
        total_score_matrix.append(list(question_avg_data_item['avg_accuracy_predict'].values))

    total_score_matrix = np.array(total_score_matrix)
    total_score_matrix = total_score_matrix.T
    
    heatmap=sns.heatmap(total_score_matrix, annot=False, cmap='Reds', fmt=".2f", linewidths=1, ax=ax2, cbar=False, xticklabels=['U','P','I','H','1a','1b','1c','2a','2b','2c','3a','3b','3c'], yticklabels=False)
    ax2.set_title('Average Post-Test Score',fontsize=14)
    ax2.set_ylabel('Student ID: 27 $\\rightarrow$ 1',fontsize=12)

    tick_labels = ['     U','     P','     I','     H','     1a','     1b','     1c','     2a','     2b','     2c','     3a','     3b','     3c']
    tick_colors = ['green','blue','teal','red','black','black','black','black','black','black','black','red','red']

    # Set the tick positions and labels for the x-axis
    x_ticks = range(len(tick_labels))

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(tick_labels)

    # Set different colors for tick labels
    for i, label in enumerate(ax2.get_xticklabels()):
        label.set_color(tick_colors[i])




    ax2.text(-1, -0.5, 'B', fontsize=20, color='black')
    heatmap.axes.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # ax2.set_xlabel('Understanding, Pre Test, ISC, Type: 1a $\\rightarrow$ 3c',fontsize=12)

    cax2 = fig.add_axes([0.93, 0.12, 0.01, 0.76])  # Adjust the position and size of the color bar
    cbar2 = plt.colorbar(ax2.get_children()[0], cax=cax2, orientation="vertical")
    cbar2.ax.tick_params(labelsize=16)
    

    plt.tight_layout()
    # plt.tight_layout(h_pad=0.5, w_pad=2)

    plt.savefig('matrix_vis.pdf')
    plt.show()


def generate_pearson_matrix_question(predict_result_path,course_name):
    predict_result_table = pd.read_csv(predict_result_path)
    predict_result_table = predict_result_table[predict_result_table['course_name']==course_name]
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
        data[type_name_dict[str(run_id)]] = list(table_item['avg_accuracy_predict'].values)
        print(run_id,data[type_name_dict[str(run_id)]])
    
    data['U'] = past_score_avg
    data['Pre'] = pre_test_list
    data['ISC'] = isc_list
    df = pd.DataFrame(data)

    correlation_matrix = df.corr()

    return correlation_matrix

def visual_pearson_matrix_question(predict_result_path,course_name):
    predict_result_table = pd.read_csv(predict_result_path)
    predict_result_table = predict_result_table[predict_result_table['course_name']==course_name]
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
        data[type_name_dict[str(run_id)]] = list(table_item['avg_accuracy_predict'].values)
        print(run_id,data[type_name_dict[str(run_id)]])
    
    data['U'] = past_score_avg
    data['Pre'] = pre_test_list
    data['ISC'] = isc_list
    df = pd.DataFrame(data)

    correlation_matrix = df.corr()

    sns.set(style="white")
    plt.figure(figsize=(10, 8))
    heatmap=sns.heatmap(correlation_matrix, annot_kws={"fontsize":14}, annot=True, cmap='Reds', fmt=".2f")
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    # plt.xlabel('X-Axis Label', fontsize=20)
    # plt.ylabel('Y-Axis Label', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)


    
    course_name_dict = {'star':'Star','birth':'Birth'}
    plt.title('Pearson Correlation Matrix - '+course_name_dict[course_name], fontsize=20)

    plt.tight_layout()

    plt.savefig('pearson_question_'+course_name_dict[course_name]+'.pdf')
    plt.show()


def visual_pearson_matrix_main(predict_result_path,predict_result_path_question,question_course):
    predict_result_table = pd.read_csv(predict_result_path)
    predict_result_birth = predict_result_table[predict_result_table['course_name']=='birth']
    predict_result_star = predict_result_table[predict_result_table['course_name']=='star']
    transcript_id_list_birth = list(set(predict_result_birth['transcript_id']))
    transcript_id_list_birth.sort()
    transcript_id_list_birth = [transcript_id_list_birth[0],transcript_id_list_birth[-1]]
    transcript_id_list_star = list(set(predict_result_star['transcript_id']))
    transcript_id_list_star.sort()
    transcript_id_list_star = [transcript_id_list_star[0],transcript_id_list_star[-1]]

    fig = plt.figure(figsize=(18,9))

    gs = GridSpec(2, 4, figure=fig)
    # gs = GridSpec(2, 7, figure=fig, width_ratios=[1, 1, 1, 1, 2, 2, 2])
    ax11 = fig.add_subplot(gs[0, 0])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax12 = fig.add_subplot(gs[0, 1])
    ax1 = [ax11,ax12]
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    ax2 = [ax21,ax22]
    ax3 = fig.add_subplot(gs[:, 2:])

    type_name_dict = {'each': 'b', 'all': 'a', 'past': 'c', 'label': 'Hu'}
    # type_name_dict = {'each': 'Type b', 'all': 'Type a', 'past': 'Type c', 'label': 'Human'}
    tick_font = 20
    
    icon_dict_birth = {0:'D',1:'E'}
    icon_dict_star = {0:'F',1:'G'}
    icon_size = 50

    for it,transcript_id in enumerate(transcript_id_list_birth):
        predict_result_table_item = predict_result_birth[predict_result_birth['transcript_id']==transcript_id]
        run_id_list = ['label','all','each','past']
        data = {}
        for run_id in run_id_list:
            table_item = predict_result_table_item[predict_result_table_item['predict_type']==run_id]
            data[type_name_dict[str(run_id)]] = list(table_item['pupil_size_predict'].values)
        df = pd.DataFrame(data)
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='Reds', annot_kws={"fontsize":16}, fmt=".2f", ax=ax1[it],cbar=False)
        ax1[it].set_title('Birth: Slide: '+str(transcript_id),fontsize=24)
        ax1[it].tick_params(axis='x', labelsize=tick_font) 
        ax1[it].tick_params(axis='y', labelsize=tick_font) 
        ax1[it].text(-0.4, -0.1, icon_dict_birth[it], fontsize=icon_size, color='black')
    
    for it,transcript_id in enumerate(transcript_id_list_star):
        predict_result_table_item = predict_result_star[predict_result_star['transcript_id']==transcript_id]
        run_id_list = ['label','all','each','past']
        data = {}
        for run_id in run_id_list:
            table_item = predict_result_table_item[predict_result_table_item['predict_type']==run_id]
            data[type_name_dict[str(run_id)]] = list(table_item['pupil_size_predict'].values)
        df = pd.DataFrame(data)
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='Reds', annot_kws={"fontsize":16}, fmt=".2f", ax=ax2[it],cbar=False)
        ax2[it].set_title('Star: Slide: '+str(transcript_id),fontsize=24)
        ax2[it].tick_params(axis='x', labelsize=tick_font) 
        ax2[it].tick_params(axis='y', labelsize=tick_font) 
        ax2[it].text(-0.4, -0.1, icon_dict_star[it], fontsize=icon_size, color='black')

    correlation_matrix_question = generate_pearson_matrix_question(predict_result_path_question,question_course)

    heatmap=sns.heatmap(correlation_matrix_question, annot_kws={"fontsize":14}, annot=True, cmap='Reds', fmt=".2f",ax=ax3)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    ax3.tick_params(axis='x', labelsize=20) 
    ax3.tick_params(axis='y', labelsize=20) 
    ax3.text(-1, -0.1, 'H', fontsize=icon_size, color='black')
    
    course_name_dict = {'star':'Star','birth':'Birth'}
    ax3.set_title('Pearson Correlation Matrix - '+course_name_dict[question_course], fontsize=20)

    # axes[4][3].set_visible(False)

    sns.set(style="white")
    plt.tight_layout()
    plt.savefig('pearson_matrix_main.pdf')
    plt.show()

def visual_pearson_matrix_node_all(predict_result_path):
    predict_result_table = pd.read_csv(predict_result_path)
    predict_result_birth = predict_result_table[predict_result_table['course_name']=='birth']
    predict_result_star = predict_result_table[predict_result_table['course_name']=='star']
    transcript_id_list_birth = list(set(predict_result_birth['transcript_id']))
    transcript_id_list_birth.sort()
    transcript_id_list_star = list(set(predict_result_star['transcript_id']))
    transcript_id_list_star.sort()

    col_n = 4
    fig, axes = plt.subplots(5, 4, figsize=(12, 16))

    type_name_dict = {'each': 'Tb', 'all': 'Ta', 'past': 'Tc', 'label': 'Hu'}
    
    for it,transcript_id in enumerate(transcript_id_list_birth):
        predict_result_table_item = predict_result_birth[predict_result_birth['transcript_id']==transcript_id]
        # run_id_list = list(set(predict_result_table_item['predict_type']))
        # run_id_list.sort()
        run_id_list = ['label','all','each','past']

        data = {}
    
        for run_id in run_id_list:
            table_item = predict_result_table_item[predict_result_table_item['predict_type']==run_id]
            data[type_name_dict[str(run_id)]] = list(table_item['pupil_size_predict'].values)

        df = pd.DataFrame(data)

        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='Reds', annot_kws={"fontsize":16}, fmt=".2f", ax=axes[int(it/col_n)][it-int(it/col_n)*col_n],cbar=False)
        axes[int(it/col_n)][it-int(it/col_n)*col_n].set_title('Birth: Slide: '+str(transcript_id),fontsize=24)
        axes[int(it/col_n)][it-int(it/col_n)*col_n].tick_params(axis='x', labelsize=20) 
        axes[int(it/col_n)][it-int(it/col_n)*col_n].tick_params(axis='y', labelsize=20) 

    
    
    for it,transcript_id in enumerate(transcript_id_list_star):
        predict_result_table_item = predict_result_star[predict_result_star['transcript_id']==transcript_id]
        # run_id_list = list(set(predict_result_table_item['predict_type']))
        # run_id_list.sort()
        run_id_list = ['label','all','each','past']

        data = {}
    
        for run_id in run_id_list:
            table_item = predict_result_table_item[predict_result_table_item['predict_type']==run_id]
            data[type_name_dict[str(run_id)]] = list(table_item['pupil_size_predict'].values)

        df = pd.DataFrame(data)

        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='Reds', annot_kws={"fontsize":16}, fmt=".2f", ax=axes[int((it+12)/col_n)][it+12-int((it+12)/col_n)*col_n],cbar=False)
        axes[int((it+12)/col_n)][it+12-int((it+12)/col_n)*col_n].set_title('Star: Slide: '+str(transcript_id),fontsize=24)
        axes[int((it+12)/col_n)][it+12-int((it+12)/col_n)*col_n].tick_params(axis='x', labelsize=20) 
        axes[int((it+12)/col_n)][it+12-int((it+12)/col_n)*col_n].tick_params(axis='y', labelsize=20) 

    axes[2][2].set_visible(False)
    axes[2][3].set_visible(False)
    axes[4][3].set_visible(False)

    sns.set(style="white")
    plt.tight_layout()
    plt.savefig('pearson_matrix_node_all.pdf')
    plt.show()

def generate_pearson_trend(predict_result_path,course_name):
    predict_result_table = pd.read_csv(predict_result_path)
    predict_result_table = predict_result_table[predict_result_table['course_name']==course_name]
    transcript_id_list = list(set(predict_result_table['transcript_id']))

    pearson_trend_list = []
    
    for it,transcript_id in enumerate(transcript_id_list):
        predict_result_table_item = predict_result_table[predict_result_table['transcript_id']==transcript_id]
        run_id_list = list(set(predict_result_table_item['predict_type']))
        run_id_list.sort()

        data = {}
    
        for run_id in run_id_list:
            table_item = predict_result_table_item[predict_result_table_item['predict_type']==run_id]
            data[str(run_id)] = list(table_item['pupil_size_predict'].values)

        df = pd.DataFrame(data)

        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == 'label' and col2 != 'label':
                    pearson_r_temp, _ = pearsonr(df[col1], df[col2])
                    pearson_trend_list.append([transcript_id,col2,round(pearson_r_temp,2)])

    pearson_trend_data = pd.DataFrame(np.array(pearson_trend_list),columns=['transcript_id','predict_type','pearson_r'])    
    pearson_trend_data['pearson_r'] = pearson_trend_data['pearson_r'].astype(float)

    return pearson_trend_data

def visual_pearson_trend(predict_result_path):
    fig,axes = plt.subplots(1,2,figsize=(11,5))
    title_dict = {0:'Birth',1:'Star'}
    pearson_trend_data = [generate_pearson_trend(predict_result_path,'birth'),generate_pearson_trend(predict_result_path,'star')]
    for i,data_item in enumerate(pearson_trend_data):
        data_item['Simulation'] = data_item['predict_type'].replace({'each': 'Type b', 'all': 'Type a', 'past': 'Type c', 'label': 'Label'})
        sns.lineplot(data=data_item,x='transcript_id',y='pearson_r',hue='Simulation',marker='o',ax=axes[i],markersize=16,linestyle='dashed',alpha=0.7)
        axes[i].set_title(title_dict[i], fontsize=20)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('Slide ID', fontsize=20)
        axes[i].set_ylabel('Pearson r', fontsize=20)
        axes[i].tick_params(axis='x', labelsize=20) 
        axes[i].tick_params(axis='y', labelsize=20) 

    sns.set(style="white")
    plt.tight_layout()
    plt.savefig('pearson_trend_slide_id.pdf')
    plt.show()


# main text figure
# visual_pearson_matrix_main('result_node.csv','result_question_avg.csv','birth') 
# matrix_question_student_grid('result_question_item.csv','datasets/student_result.csv','result_question_avg.csv','birth') 

# appendix figure
# visual_pearson_trend('result_node.csv')
# visual_pearson_matrix_node_all('result_node.csv')
# visual_pearson_matrix_question('result_question_avg.csv','star') 
matrix_question_student_grid('result_question_item.csv','datasets/student_result.csv','result_question_avg.csv','star') 