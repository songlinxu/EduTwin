from scipy.io import whosmat, loadmat
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
import openai 
import re, os, sys, time, math 
from transcript_map import transcript_config_all, post_question_dict_all
from scipy.stats import pointbiserialr

from utils import func_LLM_predict_score, func_LLM_predict_score_all, func_generate_student_demographic

def Experiment_3_node(student_result_file,student_demo_file,course_material_file,output_file,task_description_each,task_description_all,task_description_past):
    # Note: the exp_id here means the exp number in the PNAS paper: Synchronized eye movements predict test scores in online video education
    # 'start_timestamp','end_timestamp','transcript_id','course_name','exp_id','pupil_size_avg','pupil_size_rela','pupil_size_predict','predict_type','student_id','past_pupil_list'
    if os.path.exists(output_file) == False:
        with open(output_file, "a+") as file1:
            file1.write('start_timestamp,end_timestamp,transcript_id,course_name,exp_id,pupil_size_avg,pupil_size_rela,pupil_size_predict,predict_type,student_id,past_pupil_list\n')
    existing_result = pd.read_csv(output_file)
    with open(output_file, 'r') as filer:
        existing_result_string = filer.readlines()
    if len(existing_result_string) == 0:
        with open(output_file, "a+") as file1:
            file1.write('start_timestamp,end_timestamp,transcript_id,course_name,exp_id,pupil_size_avg,pupil_size_rela,pupil_size_predict,predict_type,student_id,past_pupil_list\n')

    result_table = pd.read_csv(student_result_file)
    demo_table = pd.read_csv(student_demo_file)
    course_table = pd.read_csv(course_material_file)
    exp_id_list = list(set(result_table['exp_id']))
    course_name_list = ['star','birth']
    
    study_result_list = []
    counter = 0
    limit_num = 200
    for i,exp_id in enumerate(exp_id_list):
        for j,course_name in enumerate(course_name_list):
            result_table_sub = result_table[(result_table['exp_id']==exp_id)&(result_table['course_name']==course_name)]
            student_id_list = list(set(result_table_sub['student_id']))
            transcript_id_list = list(set(result_table_sub['transcript_id']))
            transcript_id_list.sort()
            for k,student_id in enumerate(student_id_list):
                if counter > limit_num: break
                previous_result = existing_result[(existing_result['exp_id']==exp_id)&(existing_result['course_name']==course_name)&(existing_result['student_id']==student_id)]
                if len(previous_result) == 4 * len(transcript_config_all[course_name]): continue

                student_demo = func_generate_student_demographic(demo_table[(demo_table['student_id']==student_id)&(demo_table['course_name']==course_name)])
                course_content_all = ''
                print('='*50,' start all course slides simulation')
                for slide_id, slide_content in enumerate(transcript_config_all[course_name]):
                    if slide_content['id'] in transcript_id_list:
                        course_content_all = course_content_all + 'Slide ' + str(slide_content['id']) + ': ' + slide_content['content'] + '. \n'

                student_persona_all = student_demo + '. The course topic is about ' + course_name + '. The contents of each slide are shown below: \n [' + course_content_all + '].'
                output_num = len(transcript_id_list)
                pupil_size_predict_all_dict = func_LLM_predict_score_all(task_description_all,student_persona_all,'Slide',output_num)
                

                for h,transcript_id in enumerate(transcript_id_list):
                    print('='*50,' start each course slide simulation in slide ',transcript_id)
                    previous_result = existing_result[(existing_result['exp_id']==exp_id)&(existing_result['course_name']==course_name)&(existing_result['student_id']==student_id)&(existing_result['transcript_id']==transcript_id)&(existing_result['predict_type']=='each')]
                    if len(previous_result) != 0: continue
                    counter += 1
                    if counter > limit_num: break
                    result_item = result_table[(result_table['exp_id']==exp_id)&(result_table['course_name']==course_name)&(result_table['student_id']==student_id)&(result_table['transcript_id']==transcript_id)]
                    if len(result_item) != 1: 
                        print('-'*80,' error in result_item len!')
                    start_timestamp = result_item['start_timestamp'].values[0]
                    end_timestamp = result_item['end_timestamp'].values[0]
                    pupil_size_avg = result_item['pupil_size_avg'].values[0]
                    pupil_size_rela = result_item['pupil_size_rela'].values[0]
                    
                    course_content = transcript_config_all[course_name][transcript_id]['content']
                    student_persona_each = student_demo + '. The course topic is about ' + course_name + '. The current slide is slide number ' + str(int(transcript_id)) + '. The contents of the current slide is below. \n[' + course_content + '].'
                    
                    if transcript_id == 0:
                        student_persona_past = student_demo + '. The course topic is about ' + course_name + '. The contents of each slide are shown below: \n [' + course_content_all + '].' + '\n The current slide is slide number ' + str(int(transcript_id)) + '. The contents of the current slide is below. \n[' + course_content + ']. \n Since this is the first slide, there is no past understanding level for this slide.'                     
                    else:
                        student_persona_past = student_demo + '. The course topic is about ' + course_name + '. The contents of each slide are shown below: \n [' + course_content_all + '].' + '\n The current slide is slide number ' + str(int(transcript_id)) + '. The contents of the current slide is below. \n[' + course_content + ']. \n The student\'s past understanding levels for previous slides are: ' 
                        result_sub = result_table[(result_table['exp_id']==exp_id)&(result_table['course_name']==course_name)&(result_table['student_id']==student_id)&(result_table['transcript_id']<transcript_id)]
                        if len(result_sub) == 0:
                            print('-'*80,' error because len(result_sub) == 0!')
                            assert 0 == 1
                        transcript_id_past_list = list(set(result_sub['transcript_id']))
                        transcript_id_past_list.sort()
                        past_understand_level_list = []
                        for r in transcript_id_past_list:
                            result_sub_item = result_sub[result_sub['transcript_id']==r]
                            past_understand_level_list.append(1-result_sub_item['pupil_size_rela'].values[0])

                        for t,past_understand_level in enumerate(past_understand_level_list):
                            student_persona_past = student_persona_past + 'Slide ' + str(t) + ': ' + str(round(past_understand_level,2)) + ', '
                    
                    pupil_size_predict_past = func_LLM_predict_score(task_description_past,student_persona_past,'Slide')

                    pupil_size_predict_each = func_LLM_predict_score(task_description_each,student_persona_each,'Slide')

                    if transcript_id == 0:
                        past_pupil_list_str = '-'
                    else:
                        past_pupil_list_str = '-'.join([str(round(past_level,2)) for past_level in past_understand_level_list])

                

                    label_confidence = 1 - pupil_size_rela
                    with open(output_file, "a+") as file1:
                        file1.write(str(start_timestamp)+','+str(end_timestamp)+','+str(transcript_id)+','+str(course_name)+','+str(exp_id)+','+str(pupil_size_avg)+','+str(pupil_size_rela)+','+str(pupil_size_predict_each)+',each,'+str(student_id)+','+past_pupil_list_str+'\n')
                        file1.write(str(start_timestamp)+','+str(end_timestamp)+','+str(transcript_id)+','+str(course_name)+','+str(exp_id)+','+str(pupil_size_avg)+','+str(pupil_size_rela)+','+str(pupil_size_predict_all_dict[transcript_id])+',all,'+str(student_id)+','+past_pupil_list_str+'\n')
                        file1.write(str(start_timestamp)+','+str(end_timestamp)+','+str(transcript_id)+','+str(course_name)+','+str(exp_id)+','+str(pupil_size_avg)+','+str(pupil_size_rela)+','+str(pupil_size_predict_past)+',past,'+str(student_id)+','+past_pupil_list_str+'\n')
                        file1.write(str(start_timestamp)+','+str(end_timestamp)+','+str(transcript_id)+','+str(course_name)+','+str(exp_id)+','+str(pupil_size_avg)+','+str(pupil_size_rela)+','+str(label_confidence)+',label,'+str(student_id)+','+past_pupil_list_str+'\n')


def Experiment_3_question(isc_file,student_pupil_file,student_answer_file,student_demo_file,course_material_file,question_file,output_file_1,output_file_2,task_description_1,task_description_2,task_description_3,task_description_4,task_description_5,task_description_6,task_description_8,task_description_9,task_description_10):
    # 'exp_id','course_name','question_id','accuracy_per_predict','predict_type','student_id'
    # 'exp_id','course_name','label','avg_accuracy_predict','predict_type','student_id'
    if os.path.exists(output_file_1) == False:
        with open(output_file_1, "a+") as file1:
            file1.write('exp_id,course_name,question_id,accuracy_per_predict,predict_type,student_id\n')
    if os.path.exists(output_file_2) == False:
        with open(output_file_2, "a+") as file1:
            file1.write('exp_id,course_name,label,avg_accuracy_predict,predict_type,student_id,past_pupil,pre_test,isc\n')
    existing_result_1 = pd.read_csv(output_file_1)
    existing_result_2 = pd.read_csv(output_file_2)
    with open(output_file_1, 'r') as filer:
        existing_result_string_1 = filer.readlines()
    with open(output_file_2, 'r') as filer:
        existing_result_string_2 = filer.readlines()
    if len(existing_result_string_1) == 0:
        with open(output_file_1, "a+") as file1:
            file1.write('exp_id,course_name,question_id,accuracy_per_predict,predict_type,student_id\n')
    if len(existing_result_string_2) == 0:
        with open(output_file_2, "a+") as file1:
            file1.write('exp_id,course_name,label,avg_accuracy_predict,predict_type,student_id,past_pupil,pre_test,isc\n')

    isc_table = pd.read_csv(isc_file)
    pupil_table = pd.read_csv(student_pupil_file)
    result_table = pd.read_csv(student_answer_file)
    demo_table = pd.read_csv(student_demo_file)
    course_table = pd.read_csv(course_material_file)
    question_table = pd.read_csv(question_file,delimiter='\t')
    exp_id_list = list(set(result_table['exp_id']))
    # course_name_list = list(set(result_table['course_name']))
    # course_name_list = ['birth','star']
    course_name_list = ['star','birth']
    
    study_result_list = []
    counter = 0
    limit_num = 40
    for i,exp_id in enumerate(exp_id_list):
        for j,course_name in enumerate(course_name_list):
            isc_table_sub = isc_table[isc_table['course_name']==course_name]
            result_table_sub = result_table[(result_table['exp_id']==exp_id)&(result_table['course_name']==course_name)]
            student_id_list = list(set(result_table_sub['student_id']))

            pupil_table_sub = pupil_table[(pupil_table['exp_id']==exp_id)&(pupil_table['course_name']==course_name)]
            transcript_id_list = list(set(pupil_table_sub['transcript_id']))
            transcript_id_list.sort()

            question_table_post_sub = question_table[(question_table['exp_id']==exp_id)&(question_table['course_name']==course_name)&(question_table['test_type']=='post')]
            question_id_list = list(set(question_table_post_sub['question_id']))
            question_id_list.sort()
            question_content_all = '\n\n The post-test questions for students to answer are shown below: \n'
            for q, question_id_1 in enumerate(question_id_list):
                question_item_post = question_table_post_sub[question_table_post_sub['question_id']==question_id_1]
                option_contents_post = ''
                for option_id_post in [1,2,3,4]:
                    option_item_post = question_item_post[question_item_post['choice_id']==option_id_post]
                    option_contents_post = option_contents_post + 'Option ' + str(option_id_post) + ': ' + option_item_post['choice_content'].values[0] + ', '

                question_content_all = question_content_all + 'Question ' + str(question_id_1) + ': ' + question_item_post['question_content'].values[0] + '. \n This question has four options: ' + option_contents_post + '.\n'

            for k,student_id in enumerate(student_id_list):
                previous_result_1 = existing_result_1[(existing_result_1['exp_id']==exp_id)&(existing_result_1['course_name']==course_name)&(existing_result_1['student_id']==student_id)]
                if len(previous_result_1) != 0: continue
                if counter > limit_num: break
                counter += 1
                student_demo = func_generate_student_demographic(demo_table[demo_table['student_id']==student_id])
                course_content_all = ''
                for slide_id, slide_content in enumerate(transcript_config_all[course_name]):
                    if slide_content['id'] in transcript_id_list:
                        course_content_all = course_content_all + 'Slide ' + str(slide_content['id']) + ': ' + slide_content['content'] + '. \n'
                student_persona_1 = student_demo + '. \nThe course topic is about ' + course_name + '. The contents of each slide are shown below: \n [' + course_content_all + ']. \n\n' + question_content_all
                question_predict_all_1 = func_LLM_predict_score_all(task_description_1,student_persona_1,'Question',len(question_id_list))
                question_predict_total_1 = func_LLM_predict_score(task_description_4,student_persona_1,'Predicted:')

                past_confidence_all = 'The student\'s past understanding levels for previous slides are: ' 
                pupil_table_sub_sub = pupil_table[(pupil_table['exp_id']==exp_id)&(pupil_table['course_name']==course_name)&(pupil_table['student_id']==student_id)]
                if len(pupil_table_sub_sub) == 0:
                    print('-'*80,' error because len(pupil_table_sub_sub) == 0!')
                    assert 0 == 1
              
                past_understand_level_list = []
                for r in transcript_id_list:
                    pupil_table_sub_sub_item = pupil_table_sub_sub[pupil_table_sub_sub['transcript_id']==r]
                    past_understand_level_list.append(1-pupil_table_sub_sub_item['pupil_size_rela'].values[0])

                for t,past_understand_level in enumerate(past_understand_level_list):
                    past_confidence_all = past_confidence_all + 'Slide ' + str(t) + ': ' + str(round(past_understand_level,2)) + ', '

                student_persona_2 = student_demo + '. \nThe course topic is about ' + course_name + '. The contents of each slide are shown below: \n [' + course_content_all + ']. \n\n' + past_confidence_all + '\n\n ' + question_content_all
                question_predict_all_2 = func_LLM_predict_score_all(task_description_2,student_persona_2,'Question',len(question_id_list))
                question_predict_total_2 = func_LLM_predict_score(task_description_5,student_persona_2,'Predicted:')

                question_table_pre_sub = question_table[(question_table['exp_id']==exp_id)&(question_table['course_name']==course_name)&(question_table['test_type']=='pre')]
                question_pre_id_list = list(set(question_table_pre_sub['question_id']))
                question_pre_id_list.sort()
                pre_test_question_all = '\n\n  The pre-test questions which have been answered by students before the course are shown below: \n'
                for m, question_id_2 in enumerate(question_pre_id_list):
                    question_item = question_table_pre_sub[question_table_pre_sub['question_id']==question_id_2]
                    option_contents = ''
                    for option_id in [1,2,3,4]:
                        option_item = question_item[question_item['choice_id']==option_id]
                        option_contents = option_contents + 'Option ' + str(option_id) + ': ' + option_item['choice_content'].values[0] + ', '

                    pre_test_question_all = pre_test_question_all + 'Question ' + str(question_id_2) + ': ' + question_item['question_content'].values[0] + '. \n This question has four options: ' + option_contents + '.\n'

                answer_sub_pre = result_table[(result_table['exp_id']==exp_id)&(result_table['course_name']==course_name)&(result_table['test_type']=='pre')&(result_table['student_id']==student_id)]
                pre_test_score_avg = float(answer_sub_pre['score'].values[0])/100.0
                pre_test_all = pre_test_question_all + '\n The student\'s average accuracy for these pre-test questions is: ' + str(round(pre_test_score_avg,2)) + '.\n'

                student_persona_3 = student_demo + '. \nThe course topic is about ' + course_name + '. The contents of each slide are shown below: \n [' + course_content_all + ']. \n\n' + past_confidence_all + '\n\n' + pre_test_all + '\n \n' + question_content_all 
                question_predict_all_3 = func_LLM_predict_score_all(task_description_3,student_persona_3,'Question',len(question_id_list))
                question_predict_total_3 = func_LLM_predict_score(task_description_6,student_persona_3,'Predicted:')

                isc_table_sub_item = isc_table_sub[isc_table_sub['student_id']==student_id]
                isc_avg = isc_table_sub_item['isc'].values[0]
                overall_engagement = '\n The overall course engagement of this student is : ' + str(round(isc_avg,2)) + ' (0 is lowest engagement and 1 is highest engagement). \n'

                question_predict_ind_2 = {}
                question_predict_ind_3 = {}
                question_predict_ind_4 = {}

                for q_q, question_id_1_1 in enumerate(question_id_list):
                    question_item_post = question_table_post_sub[question_table_post_sub['question_id']==question_id_1_1]
                    option_contents_post = ''
                    for option_id_post in [1,2,3,4]:
                        option_item_post = question_item_post[question_item_post['choice_id']==option_id_post]
                        option_contents_post = option_contents_post + 'Option ' + str(option_id_post) + ': ' + option_item_post['choice_content'].values[0] + ', '

                    current_engagement = overall_engagement
                    current_course_content = ''
                    current_confidence = 'The student understanding level for these current slides are: '
                    for slide_id_q in post_question_dict_all[course_name][question_id_1_1]['slide_id_list']:
                        current_course_content = current_course_content + 'Slide: '+str(slide_id_q)+': '+transcript_config_all[course_name][slide_id_q]['content'] + '. \n' 
                        current_confidence = current_confidence + 'Slide: ' + str(round(past_understand_level_list[slide_id_q],2)) + ', '
                    current_confidence = current_confidence +'(0 is lowest level and 1 is highest level).'
                    current_post_question = 'The current post question is: Question-' + str(question_id_1_1) + ': ' + question_item_post['question_content'].values[0] + '. \n This question has four options: ' + option_contents_post + '.\n' + '.\n Please predict whether the student could answer this Question-'+str(question_id_1_1)+' correctly or not.'
                    student_persona_5 = student_demo + '. \nThe course topic is about ' + course_name + '. The contents of the current slides are shown below: \n [' + current_course_content + ']. \n\n' + current_confidence + '\n \n' + current_post_question 
                    student_persona_6 = student_demo + '. \nThe course topic is about ' + course_name + '. The contents of the current slides are shown below: \n [' + current_course_content + ']. \n\n' + pre_test_all + '\n \n' + current_confidence + '\n \n' + current_post_question 
                    student_persona_7 = student_demo + '. \nThe course topic is about ' + course_name + '. The contents of the current slides are shown below: \n [' + current_course_content + ']. \n\n' + pre_test_all + '\n \n' + current_confidence + '\n \n' + current_engagement + '\n \n' + current_post_question 
                    question_predict_ind_2[question_id_1_1] = func_LLM_predict_score(task_description_8,student_persona_5,'Question')
                    question_predict_ind_3[question_id_1_1] = func_LLM_predict_score(task_description_9,student_persona_6,'Question')
                    question_predict_ind_4[question_id_1_1] = func_LLM_predict_score(task_description_10,student_persona_7,'Question')


                for h,question_id in enumerate(question_id_list):
                    with open(output_file_1, "a+") as file1: 
                        # exp_id,course_name,question_id,accuracy_per_predict,student_id
                        file1.write(str(exp_id)+','+str(course_name)+','+str(question_id)+','+str(question_predict_all_1[question_id])+',type_1_only_course,'+str(student_id)+'\n')
                        file1.write(str(exp_id)+','+str(course_name)+','+str(question_id)+','+str(question_predict_all_2[question_id])+',type_2_course_confidence,'+str(student_id)+'\n')
                        file1.write(str(exp_id)+','+str(course_name)+','+str(question_id)+','+str(question_predict_all_3[question_id])+',type_3_course_confidence_pretest,'+str(student_id)+'\n')
                        file1.write(str(exp_id)+','+str(course_name)+','+str(question_id)+','+str(question_predict_ind_2[question_id])+',type_2_ind_course_confidence,'+str(student_id)+'\n')
                        file1.write(str(exp_id)+','+str(course_name)+','+str(question_id)+','+str(question_predict_ind_3[question_id])+',type_3_ind_course_confidence_pretest,'+str(student_id)+'\n')
                        file1.write(str(exp_id)+','+str(course_name)+','+str(question_id)+','+str(question_predict_ind_4[question_id])+',type_4_ind_course_confidence_pretest_isc,'+str(student_id)+'\n')


                answer_sub_post = result_table[(result_table['exp_id']==exp_id)&(result_table['course_name']==course_name)&(result_table['test_type']=='post')&(result_table['student_id']==student_id)]
                if len(answer_sub_post) != 1:
                    print('-'*80,' error because len(answer_sub_post) != 1!')
                    assert 0 == 1
                
                past_understand_level_str = '-'.join([str(round(pu_i,2)) for pu_i in past_understand_level_list])

                with open(output_file_2, "a+") as file2: 
                    # exp_id,course_name,label,avg_accuracy_predict,predict_type,student_id
                    label_value = float(answer_sub_post['score'].values[0])/100
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(label_value)+','+'label'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(np.mean(list(question_predict_all_1.values())))+','+'type_1_only_course'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(np.mean(list(question_predict_all_2.values())))+','+'type_2_course_confidence'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(np.mean(list(question_predict_all_3.values())))+','+'type_3_course_confidence_pretest'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(question_predict_total_1)+','+'type_1_total_only_course'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(question_predict_total_2)+','+'type_2_total_course_confidence'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(question_predict_total_3)+','+'type_3_total_course_confidence_pretest'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(np.mean(list(question_predict_ind_2.values())))+','+'type_2_ind_course_confidence'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(np.mean(list(question_predict_ind_3.values())))+','+'type_3_ind_course_confidence_pretest'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
                    file2.write(str(exp_id)+','+str(course_name)+','+str(label_value)+','+str(np.mean(list(question_predict_ind_4.values())))+','+'type_4_ind_course_confidence_pretest_isc'+','+str(student_id)+','+past_understand_level_str+','+str(pre_test_score_avg)+','+str(isc_avg)+'\n')
   
# task 1: understanding level simulation
# task_description_each = 'Different backgrounds of students in different stages of the course may result in different student understanding level of the current slide. As slides move on, students may also get tired and absent-minded during the course. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background and the contents of a specific slide below. You should give me a numeric score from 0 to 1 to estimate to what extent the student could understand the current slide according to three factors: student background, slide contents and slide number. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input the student background and course contents below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Slide-ID: your predicted score.'
# task_description_all = 'Different backgrounds of students in different stages of the course may result in different student understanding level of the current slide. As slides move on, students may also get tired and absent-minded during the course. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background and the contents of all slides in a course below. For each slide, you should give me a numeric score from 0 to 1 to estimate to what extent the student could understand the current slide according to three factors: student background, slide contents and slide number. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input the student background and course contents below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Slide-ID: your predicted score.'
# task_description_past = 'Different backgrounds of students in different stages of the course may result in different student understanding level of the current slide. As slides move on, students may also get tired and absent-minded during the course. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background and the contents of all slides in a course below. For each slide, you should give me a numeric score from 0 to 1 to estimate to what extent the student could understand the current slide according to four factors: student background, current slide contents, current slide number, and the student\'s past understanding levels in previous slides. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input the student background, course contents, and past understanding levels below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Slide-ID: your predicted score.'
# Experiment_3_node('datasets/student_result.csv','datasets/student_demo.csv','datasets/course_material.csv','result_node.csv',task_description_each,task_description_all,task_description_past)


# task 2: learning outcome simulation
# task_description_1 = 'Student learning performance will be affected by different factors such as course materials and student background. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, course contents, and post-test questions for the student to answer after the course to evaluate the learning performance. You should give me a numeric value of either 0 or 1 to predict whether student could correctly answer each question in the post-test according to student background, course materials, and post-test questions. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Question-ID: 0 or 1.'
# task_description_4 = 'Student learning performance will be affected by different factors such as course materials and student background. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, course contents, and post-test questions for the student to answer after the course to evaluate the learning performance. You should give me a numeric value between 0 and 1 to predict the average accuracy of the student to answer these questions in the post-test according to student background, course materials, and post-test questions. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Predicted: accuracy.'
# task_description_2 = 'Student learning performance will be affected by different factors such as course materials, student background and student understanding level of the course. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, course contents, student understanding level (ranging from 0 to 1) for each slide in the course, and post-test questions for the student to answer after the course to evaluate the learning performance. You should give me a numeric value of either 0 or 1 to predict whether student could correctly answer each question in the post-test according to student background, course materials, student understanding level, and post-test questions. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Question-ID: 0 or 1.'
# task_description_5 = 'Student learning performance will be affected by different factors such as course materials, student background and student understanding level of the course. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, course contents, student understanding level (ranging from 0 to 1) for each slide in the course, and post-test questions for the student to answer after the course to evaluate the learning performance. You should give me a numeric value of between 0 and 1 to predict the average accuracy of the student to answer these questions in the post-test according to student background, course materials, student understanding level, and post-test questions. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Predicted: accuracy.'
# task_description_8 = 'Student learning performance will be affected by different factors such as course materials, student background and student understanding level of the course. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, course contents for the current slides, student understanding level (ranging from 0 to 1) for the current slides in the course, and post-test question related with these slides for the student to answer after the course to evaluate the learning performance. You should give me a numeric value of either 0 or 1 to predict whether the student can answer such question correctly in the post-test according to student background, course materials, student understanding level, and post-test question. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Question-ID: 0 or 1.'
# task_description_3 = 'Student learning performance will be affected by different factors such as course materials, student background, student understanding level of the course, and student pre-test accuracy before the course to reflect the basic knowledge background. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, pre-test performance before the course, course contents, student understanding level (ranging from 0 to 1) for each slide in the course, and post-test questions for the student to answer after the course to evaluate the learning performance. You should give me a numeric value of either 0 or 1 to predict whether student could correctly answer each question in the post-test according to student background, pre-test score, course materials, student understanding level, and post-test questions. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Question-ID: 0 or 1.'
# task_description_6 = 'Student learning performance will be affected by different factors such as course materials, student background, student understanding level of the course, and student pre-test accuracy before the course to reflect the basic knowledge background. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, pre-test performance before the course, course contents, student understanding level (ranging from 0 to 1) for each slide in the course, and post-test questions for the student to answer after the course to evaluate the learning performance. You should give me a numeric value between 0 and 1 to predict the average accuracy of the student to answer these questions in the post-test according to student background, pre-test score, course materials, student understanding level, and post-test questions. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Predicted: accuracy.'
# task_description_9 = 'Student learning performance will be affected by different factors such as course materials, student background, student understanding level of the course, and student pre-test accuracy before the course to reflect the basic knowledge background. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, pre-test performance before the course, course contents in the current slide, student understanding level (ranging from 0 to 1) for the current slides in the course, and post-test question related with these slides for the student to answer after the course to evaluate the learning performance. You should give me a numeric value of either 0 or 1 to predict whether the student can answer such question correctly in the post-test according to student background, pre-test score, course materials, student understanding level, and post-test questions. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Question-ID: 0 or 1.'
# task_description_10 = 'Student learning performance will be affected by different factors such as course materials, student background, student understanding level of the course, student course engagement, and student pre-test accuracy before the course to reflect the basic knowledge background. Your task is to mimic a virtual student with specific background and receive the course contents like the student. I will input the student background, pre-test performance before the course, course contents in the current slide, student understanding level (ranging from 0 to 1) for the current slides in the course, average student engagement, and post-test question related with these slides for the student to answer after the course to evaluate the learning performance. You should give me a numeric value of either 0 or 1 to predict whether the student can answer such question correctly in the post-test according to student background, pre-test score, course materials, student understanding level, course engagement, and post-test questions. Note that I know different factors may contribute to different student learning but your task is to make such prediction based on my given information. If you think you understand, I will input all data below. Note that you just need to give me the numeric score according to your understanding. You do not need to give me reasons. The output format should exactly be: Question-ID: 0 or 1.'
# Experiment_3_question('datasets/isc.csv','datasets/student_result.csv','datasets/student_answer.csv','datasets/student_demo.csv','datasets/course_material.csv','datasets/student_question.csv','result_question_item.csv','result_question_avg.csv',task_description_1,task_description_2,task_description_3,task_description_4,task_description_5,task_description_6,task_description_8,task_description_9,task_description_10)
