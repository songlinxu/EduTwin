import numpy as np 
import pandas as pd 
import seaborn as sns
import geopandas as gpd
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,re 

from utils import generate_persona, func_LLM_predict_score, _get_course_info


def experiment_2_run(output_file):
    if os.path.exists(output_file) == False:
        with open(output_file, "a+") as file1:
            file1.write('course_id,course_year,student_id,past_scores,score_type,score_item\n')
    existing_result = pd.read_csv(output_file)
    with open(output_file, 'r') as filer:
        existing_result_string = filer.readlines()
    if len(existing_result_string) == 0:
        with open(output_file, "a+") as file1:
            file1.write('course_id,course_year,student_id,past_scores,score_type,score_item\n')

    student_score_file = 'datasets/OULA/studentAssessment.csv'
    student_score_data = pd.read_csv(student_score_file)

    student_info_file = 'datasets/OULA/studentInfo.csv'
    student_info_data = pd.read_csv(student_info_file)

    course_info_dict,course_list = _get_course_info()
    course_list = ['CCC','DDD']
    course_list.sort()
    predict_result_all = []
    user_limit = 100
    for i,course_id in enumerate(course_list):
        if len(predict_result_all) > user_limit: break
        course_year_info = course_info_dict[course_id]
        course_year_list = list(course_year_info.keys())
        course_year_list.sort()
        for j,course_year in enumerate(course_year_list):
            if len(predict_result_all) > user_limit: break
            assess_info_list = course_year_info[course_year]
            # print(assess_info_list)
            assess_test_id_list = [d[0] for d in assess_info_list if d[1] != 'Exam']
            # check if there is only one exam id
            assess_exam_id_list = [d[0] for d in assess_info_list if d[1] == 'Exam' ]
            if len(assess_exam_id_list) == 0:
                print('-'*40,'error because len(assess_exam_id_list) != 1, len(assess_exam_id_list) =',len(assess_exam_id_list),' in course id ',course_id, ' course year ',course_year)
                assert 1 ==0 

            # there may be more than one exam but we just use the first one.
            assess_exam_id = assess_exam_id_list[0]

            student_sub_data = student_info_data[(student_info_data['code_module']==course_id)&(student_info_data['code_presentation']==course_year)]
            student_list = list(set(student_sub_data['id_student']))
            student_list.sort()
            print(course_id,course_year,' student num: ',len(student_list))

            for k,student_id in enumerate(student_list):
                previous_result = existing_result[(existing_result['course_id']==course_id)&(existing_result['course_year']==course_year)&(existing_result['student_id']==student_id)]
                if len(previous_result) != 0: continue
                if len(predict_result_all) > user_limit: break
                # print(student_id)
                student_item = student_sub_data[student_sub_data['id_student']==student_id]
                student_assess_item = student_score_data[student_score_data['id_student']==student_id]
                student_assess_exaxm_item = student_assess_item[student_assess_item['id_assessment']==assess_exam_id]
                if len(student_assess_exaxm_item) == 0: 
                    continue
                
                if len(student_item)!=1: 
                    print('-'*40,'error!!! in student info.')
                    assert 0 == 1

                student_assess_score_list = []
                for h,access_id in enumerate(assess_test_id_list):
                    student_assess_test_item = student_assess_item[student_assess_item['id_assessment']==access_id]
                    if len(student_assess_test_item) == 0: 
                        continue
                    student_assess_score_list.append(student_assess_test_item['score'].values[0])

                if len(student_assess_score_list) < 5: 
                    # print('-'*40,'error because: len(student_assess_score_list) < 5',student_id)
                    continue

                student_persona = generate_persona(student_item)

                student_persona_with_score = student_persona + '. This student has finished previous assessments with score of '
                student_past_score = 'This student has finished previous assessments with score of '
                for s,score_each in enumerate(student_assess_score_list[:-1]):
                    student_persona_with_score = student_persona_with_score + str(score_each) + ', '
                    student_past_score = student_past_score + str(score_each) + ', '
                student_persona_with_score = student_persona_with_score + str(student_assess_score_list[-1]) + ', respectively.'
                student_past_score = student_past_score + str(student_assess_score_list[-1]) + ', respectively.'

                student_past_score_dict = {}
                predict_score_past_dict = {}
                for si_index in range(5):
                    student_past_score_dict[si_index+1] = 'This student has finished previous assessments with score of '
                    for s,score_each in enumerate(student_assess_score_list[:si_index+1]):
                        student_past_score_dict[si_index+1] = student_past_score_dict[si_index+1] + str(score_each) + ', '
                    student_past_score_dict[si_index+1] = student_past_score_dict[si_index+1] + 'respectively.'
                    predict_score_past_dict[si_index+1] = func_LLM_predict_score(student_past_score_dict[si_index+1])

                predict_score_both = func_LLM_predict_score(student_persona_with_score)
                predict_score_persona = func_LLM_predict_score(student_persona)
                predict_score_past = func_LLM_predict_score(student_past_score)
                
                label_score = student_assess_exaxm_item['score'].values[0]

                past_scores = '-'.join([str(ds) for ds in student_assess_score_list])

                with open(output_file, "a+") as file1:
                    # file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+','+str(label_score)+','+str(predict_score_both)+','+str(predict_score_persona)+','+str(predict_score_past)+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',label_score,'+str(label_score)+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',predict_score_both,'+str(predict_score_both)+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',predict_score_persona,'+str(predict_score_persona)+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',predict_score_past,'+str(predict_score_past)+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',predict_score_past_1,'+str(predict_score_past_dict[1])+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',predict_score_past_2,'+str(predict_score_past_dict[2])+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',predict_score_past_3,'+str(predict_score_past_dict[3])+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',predict_score_past_4,'+str(predict_score_past_dict[4])+'\n')
                    file1.write(str(course_id)+','+str(course_year)+','+str(student_id)+','+str(past_scores)+',predict_score_past_5,'+str(predict_score_past_dict[5])+'\n')

                predict_result_all.append([course_id,course_year,student_id,past_scores,label_score,predict_score_both])




experiment_2_run('result.csv')

