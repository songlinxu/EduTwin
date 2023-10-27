import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,math 


dataset_config = {
    'age': {
        1: '18 to 21 years old',
        2: '22 to 25 years old',
        3: 'more than 26 years old',
    },
    'gender': {
        1: 'female',
        2: 'male',
    },
    'highschool': {
        1: 'private high-school',
        2: 'state high-school',
        3: 'high-school other than private or state high-school',
    },
    'scholarship': {
        1: 'No scholarship',
        2: '25% scholarship',
        3: '50% scholarship',
        4: '75% scholarship',
        5: 'Full scholarship',
    },
    'work': {
        1: 'works while studying',
        2: 'does not work while studying',
    },
    'activity': {
        1: 'engages in regular artistic or sports activities',
        2: 'does not engage in regular artistic or sports activities',
    },
    'partner': {
        1: 'has a partner',
        2: 'does not have a partner',
    },
    'salary': {
        1: 'from 135 to 200 USD',
        2: 'from 201 to 270 USD',
        3: 'from 271 to 340 USD',
        4: 'from 341 to 410 USD',
        5: 'above 410 USD',
    },
    'transport': {
        1: 'takes the bus to the university',
        2: 'uses a private car or taxi to the university',
        3: 'rides a bicycle to the university',
        4: 'uses other transportation than bus/car/bicycle to the university',
    },
    'living': {
        1: 'rents accommodation in Cyprus',
        2: 'lives in a dormitory',
        3: 'lives with family in Cyprus',
        4: 'lives in other places instead of renting, dormitory or living with family in Cyprus',
    },
    'mother_edu': {
        1: 'mother has a primary school education',
        2: 'mother has a secondary school education',
        3: 'mother has a high school education',
        4: 'mother has a university degree',
        5: 'mother has a Master of Science degree',
        6: 'mother has a Ph.D. degree',
    },
    'father_edu': {
        1: 'father has a primary school education',
        2: 'father has a secondary school education',
        3: 'father has a high school education',
        4: 'father has a university degree',
        5: 'father has a Master of Science degree',
        6: 'father has a Ph.D. degree',
    },
    'sibling_num': {
        1: 'has 1 sibling',
        2: 'has 2 siblings',
        3: 'has 3 siblings',
        4: 'has 4 siblings',
        5: 'has 5 or more siblings',
    },
    'parental_status': {
        1: 'parents are married',
        2: 'parents are divorced',
        3: 'one or both parents have passed away',
    },
    'mother_job': {
        1: 'mother is retired',
        2: 'mother is a housewife',
        3: 'mother is a government officer',
        4: 'mother works in the private sector',
        5: 'mother is self-employed',
        6: 'mother has other occupation than retiring, housewife, government officer, private sector, or self-employment',
    },
    'father_job': {
        1: 'father is retired',
        2: 'father is a government officer',
        3: 'father works in the private sector',
        4: 'father is self-employed',
        5: 'father has other occupation than retiring, government officer, private sector, or self-employment',
    },
    'study_hour': {
        1: 'has no weekly study hours',
        2: 'has less than 5 hours of weekly study',
        3: 'has 6-10 hours of weekly study',
        4: 'has 11-20 hours of weekly study',
        5: 'has more than 20 hours of weekly study',
    },
    'read_freq_no_sci': {
        1: 'has no reading of non-scientific books/journals',
        2: 'sometimes reads non-scientific books/journals',
        3: 'often reads non-scientific books/journals',
    },
    'read_freq_sci': {
        1: 'has no reading of scientific books/journals',
        2: 'sometimes reads scientific books/journals',
        3: 'often reads scientific books/journals',
    },
    'attend_dept': {
        1: 'attends seminars/conferences related to the department',
        2: 'does not attend seminars/conferences related to the department',
    },
    'impact_project': {
        1: 'has a positive attitude on the impact of the projects/activities on the success',
        2: 'has a negative attitude on the impact of the projects/activities on the success',
        3: 'has a neutral attitude on the impact of the projects/activities on the success',
    },
    'attend_class': {
        1: 'always attends classes',
        2: 'sometimes attends classes',
        3: 'never attends classes',
    },
    'prep_study': {
        1: 'usually prepares for midterm exams alone',
        2: 'usually prepares for midterm exams with friends',
        3: '',
    },
    'prep_exam': {
        1: 'usually prepares for exams closest to the exam date',
        2: 'usually regularly prepares for exams during the semester',
        3: 'usually does not prepare for exams',
    },
    'note': {
        1: 'never takes notes in classes',
        2: 'sometimes takes notes in classes',
        3: 'always takes notes in classes',
    },
    'listen': {
        1: 'never actively listens in classes',
        2: 'sometimes actively listens in classes',
        3: 'always actively listens in classes',
    },
    'discuss': {
        1: 'thinks that the discussion never improves the interest and success in the course',
        2: 'thinks that the discussion sometimes improves the interest and success in the course',
        3: 'thinks that the discussion always improves the interest and success in the course',
    },
    'classroom': {
        1: 'finds the flip-classroom not useful',
        2: 'finds the flip-classroom useful',
        3: 'has not experienced the flip-classroom',
    },
    'cuml_gpa': {
        1: 'cumulative GPA is below 2.00',
        2: 'cumulative GPA is between 2.00-2.49',
        3: 'cumulative GPA is between 2.50-2.99',
        4: 'cumulative GPA is between 3.00-3.49',
        5: 'cumulative GPA is above 3.49',
    },
    'exp_gpa': {
        1: 'expected Cumulative GPA is below 2.00',
        2: 'expected Cumulative GPA is between 2.00-2.49',
        3: 'expected Cumulative GPA is between 2.50-2.99',
        4: 'expected Cumulative GPA is between 3.00-3.49',
        5: 'expected Cumulative GPA is above 3.49',
    },
    'course_id': {
        1: 'Assessment and Evaluation in Education',
        2: 'other courses',
        3: 'other courses',
        4: 'other courses',
        5: 'other courses',
        6: 'other courses',
        7: 'other courses',
        8: 'Electronics',
        9: 'General Physics II',
    },
    'grade': {
        0: 'Fail',
        1: 'DD',
        2: 'DC',
        3: 'CC',
        4: 'CB',
        5: 'BB',
        6: 'BA',
        7: 'AA',
    }
}

dataset_config_label = {

    'age': {
        1: '18-21',
        2: '22-25',
        3: '>26',
    },
    'gender': {
        1: 'female',
        2: 'male',
    },
    'highschool': {
        1: 'private school',
        2: 'state school',
        3: 'others',
    },
    'scholarship': {
        1: 'No',
        2: '25%',
        3: '50%',
        4: '75%',
        5: 'Full',
    },
    'work': {
        1: 'work+study',
        2: 'only study',
    },
    'activity': {
        1: 'regular activities',
        2: 'no activities',
    },
    'partner': {
        1: 'partner',
        2: 'no partner',
    },
    'salary': {
        1: '135-200USD',
        2: '201-270USD',
        3: '271-340USD',
        4: '341-410USD',
        5: '>410USD',
    },
    'transport': {
        1: 'bus',
        2: 'car/taxi',
        3: 'bicycle',
        4: 'others',
    },
    'living': {
        1: 'rent',
        2: 'dormitory',
        3: 'family',
        4: 'others',
    },
    'mother_edu': {
        1: 'primary',
        2: 'secondary',
        3: 'high',
        4: 'bachelor',
        5: 'MSc',
        6: 'PhD',
    },
    'father_edu': {
        1: 'primary',
        2: 'secondary',
        3: 'high',
        4: 'bachelor',
        5: 'MSc',
        6: 'PhD',
    },
    'sibling_num': {
        1: '1',
        2: '2s',
        3: '3',
        4: '4',
        5: '5 or more',
    },
    'parental_status': {
        1: 'married',
        2: 'divorced',
        3: 'passed away',
    },
    'mother_job': {
        1: 'retired',
        2: 'housewife',
        3: 'officer',
        4: 'private sector',
        5: 'self-employed',
        6: 'others',
    },
    'father_job': {
        1: 'retired',
        2: 'officer',
        3: 'private sector',
        4: 'self-employed',
        5: 'others',
    },
    'study_hour': {
        1: '0',
        2: '<5',
        3: '6-10',
        4: '11-20',
        5: '>20',
    },
    'read_freq_no_sci': {
        1: 'never',
        2: 'sometimes',
        3: 'often',
    },
    'read_freq_sci': {
        1: 'never',
        2: 'sometimes',
        3: 'often',
    },
    'attend_dept': {
        1: 'yes',
        2: 'no',
    },
    'impact_project': {
        1: 'positive',
        2: 'negative',
        3: 'neutral',
    },
    'attend_class': {
        1: 'always',
        2: 'sometimes',
        3: 'never',
    },
    'prep_study': {
        1: 'alone',
        2: 'with friends',
        3: 'others',
    },
    'prep_exam': {
        1: 'closest to exam',
        2: 'regularly prepare',
        3: 'no prepare',
    },
    'note': {
        1: 'never',
        2: 'sometimes',
        3: 'always',
    },
    'listen': {
        1: 'never',
        2: 'sometimes',
        3: 'always',
    },
    'discuss': {
        1: 'never',
        2: 'sometimes',
        3: 'always',
    },
    'classroom': {
        1: 'not useful',
        2: 'useful',
        3: 'no experience',
    },
    'cuml_gpa': {
        1: '<2.00',
        2: '2.00-2.49',
        3: '2.50-2.99',
        4: '3.00-3.49',
        5: '>3.49',
    },
    'exp_gpa': {
        1: '<2.00',
        2: '2.00-2.49',
        3: '2.50-2.99',
        4: '3.00-3.49',
        5: '>3.49',
    },
    'course_id': {
        1: 'Assessment and Evaluation in Education',
        2: 'other courses',
        3: 'other courses',
        4: 'other courses',
        5: 'other courses',
        6: 'other courses',
        7: 'other courses',
        8: 'Electronics',
        9: 'General Physics II',
    },
    'grade': {
        0: 'Fail',
        1: 'DD',
        2: 'DC',
        3: 'CC',
        4: 'CB',
        5: 'BB',
        6: 'BA',
        7: 'AA',
    }
}

dataset_config_label_draw = {

    'age': {
        1: '18-21 years old',
        2: '22-25 years old',
        3: '>26 years old',
    },
    'gender': {
        1: 'female',
        2: 'male',
    },
    'highschool': {
        1: 'private school',
        2: 'state school',
        3: 'others',
    },
    'scholarship': {
        1: 'No scholarship',
        2: '25% scholarship',
        3: '50% scholarship',
        4: '75% scholarship',
        5: 'Full scholarship',
    },
    'work': {
        1: 'work+study',
        2: 'only study',
    },
    'activity': {
        1: 'regular activities',
        2: 'no activities',
    },
    'partner': {
        1: 'partner',
        2: 'no partner',
    },
    'salary': {
        1: '135-200USD',
        2: '201-270USD',
        3: '271-340USD',
        4: '341-410USD',
        5: '>410USD',
    },
    'transport': {
        1: 'bus',
        2: 'car/taxi',
        3: 'bicycle',
        4: 'others',
    },
    'living': {
        1: 'rent',
        2: 'dormitory',
        3: 'family',
        4: 'others',
    },
    'mother_edu': {
        1: 'primary',
        2: 'secondary',
        3: 'high',
        4: 'bachelor',
        5: 'MSc',
        6: 'PhD',
    },
    'father_edu': {
        1: 'primary',
        2: 'secondary',
        3: 'high',
        4: 'bachelor',
        5: 'MSc',
        6: 'PhD',
    },
    'sibling_num': {
        1: '1',
        2: '2s',
        3: '3',
        4: '4',
        5: '5 or more',
    },
    'parental_status': {
        1: 'married',
        2: 'divorced',
        3: 'passed away',
    },
    'mother_job': {
        1: 'retired',
        2: 'housewife',
        3: 'officer',
        4: 'private sector',
        5: 'self-employed',
        6: 'others',
    },
    'father_job': {
        1: 'retired',
        2: 'officer',
        3: 'private sector',
        4: 'self-employed',
        5: 'others',
    },
    'study_hour': {
        1: '0',
        2: '<5',
        3: '6-10',
        4: '11-20',
        5: '>20',
    },
    'read_freq_no_sci': {
        1: 'never',
        2: 'sometimes',
        3: 'often',
    },
    'read_freq_sci': {
        1: 'never',
        2: 'sometimes',
        3: 'often',
    },
    'attend_dept': {
        1: 'yes',
        2: 'no',
    },
    'impact_project': {
        1: 'positive',
        2: 'negative',
        3: 'neutral',
    },
    'attend_class': {
        1: 'always',
        2: 'sometimes',
        3: 'never',
    },
    'prep_study': {
        1: 'alone',
        2: 'with friends',
        3: 'others',
    },
    'prep_exam': {
        1: 'closest to exam',
        2: 'regularly prepare',
        3: 'no prepare',
    },
    'note': {
        1: 'never',
        2: 'sometimes',
        3: 'always',
    },
    'listen': {
        1: 'never',
        2: 'sometimes',
        3: 'always',
    },
    'discuss': {
        1: 'never',
        2: 'sometimes',
        3: 'always',
    },
    'classroom': {
        1: 'not useful',
        2: 'useful',
        3: 'no experience',
    },
    'cuml_gpa': {
        1: '<2.00',
        2: '2.00-2.49',
        3: '2.50-2.99',
        4: '3.00-3.49',
        5: '>3.49',
    },
    'exp_gpa': {
        1: '<2.00',
        2: '2.00-2.49',
        3: '2.50-2.99',
        4: '3.00-3.49',
        5: '>3.49',
    },
    'course_id': {
        1: 'Assessment and Evaluation in Education',
        2: 'other courses',
        3: 'other courses',
        4: 'other courses',
        5: 'other courses',
        6: 'other courses',
        7: 'other courses',
        8: 'Electronics',
        9: 'General Physics II',
    },
    'grade': {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
    }
}

def generate_persona(table_item):
    # student_id,age,gender,highschool,scholarship,work,activity,partner,salary,transport,living,mother_edu,father_edu,sibling_num,parental_status,mother_job,father_job,study_hour,read_freq_no_sci,read_freq_sci,attend_dept,impact_project,attend_class,prep_study,prep_exam,note,listen,discuss,classroom,cuml_gpa,exp_gpa,course_id,grade
    header_list = table_item.columns.values
    arr_item = np.array(table_item)[0]
    pron_name = 'he' if table_item['gender'].values[0] == 2 else 'she'
    pron_whos = 'his' if table_item['gender'].values[0] == 2 else 'her'
    persona = arr_item[0] + ' is a ' + dataset_config['age'][int(table_item['age'].values[0])] + \
        ' ' + dataset_config['gender'][int(table_item['gender'].values[0])] + ' university student with ' + \
        dataset_config['scholarship'][int(table_item['scholarship'].values[0])] + ', who ' + \
        'was from a ' + dataset_config['highschool'][int(table_item['highschool'].values[0])] + '. ' + \
        pron_name + ' also ' + dataset_config['work'][int(table_item['work'].values[0])] + ' and ' + \
        dataset_config['activity'][int(table_item['activity'].values[0])] + '. Currently ' + pron_name + ' ' + \
        dataset_config['partner'][int(table_item['partner'].values[0])] + ' and has a salary ' + \
        dataset_config['salary'][int(table_item['salary'].values[0])] + '. ' + pron_name + ' ' + \
        dataset_config['living'][int(table_item['living'].values[0])] + ' and ' + \
        dataset_config['transport'][int(table_item['transport'].values[0])] + '. ' + \
        pron_whos + ' ' + dataset_config['mother_edu'][int(table_item['mother_edu'].values[0])] + \
        ' and ' + pron_whos + ' ' + dataset_config['father_edu'][int(table_item['father_edu'].values[0])] + '. ' + \
        ' In addition, ' + pron_name + ' ' + dataset_config['sibling_num'][int(table_item['sibling_num'].values[0])] + ' and ' + \
        pron_whos + ' ' + dataset_config['parental_status'][int(table_item['parental_status'].values[0])] + '. Regarding to the job of the parents, ' + \
        pron_whos + ' ' + dataset_config['mother_job'][int(table_item['mother_job'].values[0])] + ' and ' + \
        pron_whos + ' ' + dataset_config['father_job'][int(table_item['father_job'].values[0])] + '. ' + \
        pron_name + ' ' + dataset_config['study_hour'][int(table_item['study_hour'].values[0])] + ' and ' + \
        pron_name + ' ' + dataset_config['read_freq_no_sci'][int(table_item['read_freq_no_sci'].values[0])] + ' and ' + \
        pron_name + ' ' + dataset_config['read_freq_sci'][int(table_item['read_freq_sci'].values[0])] + '. Moreover, ' + \
        pron_name + ' ' + dataset_config['attend_dept'][int(table_item['attend_dept'].values[0])] + ' and ' + \
        pron_name + ' ' + dataset_config['attend_class'][int(table_item['attend_class'].values[0])] + ' and ' + \
        pron_name + ' ' + dataset_config['impact_project'][int(table_item['impact_project'].values[0])] + '. For the exam, ' + \
        pron_name + ' ' + dataset_config['prep_study'][int(table_item['prep_study'].values[0])] + ' and ' + \
        pron_name + ' ' + dataset_config['prep_exam'][int(table_item['prep_exam'].values[0])] + '. In addition, ' + \
        pron_name + ' ' + dataset_config['note'][int(table_item['note'].values[0])] + ' and ' + \
        pron_name + ' ' + dataset_config['listen'][int(table_item['listen'].values[0])] + '. What\'s more, ' + \
        pron_name + ' ' + dataset_config['discuss'][int(table_item['discuss'].values[0])] + ' and ' + \
        pron_name + ' ' + dataset_config['classroom'][int(table_item['classroom'].values[0])] + '. '
    return persona


class LLM_Assistent_Predict():
    def __init__(self):
        openai.api_key = 'Your OpenAI Key'
        self.messages = [ {"role": "system", "content":"You are an intelligent assistant."} ]

    def _interaction(self,input_text):
        self.messages.append({"role": "user", "content": input_text},)
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0, messages=self.messages)
        reply = chat.choices[0].message.content
        print('\n input text: \n',input_text)
        print('\n LLM answer: \n',reply)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _extract_prediction(self,predict_text):
        if 'Fail' in predict_text: return '0'
        if 'DD' in predict_text: return '1'
        if 'DC' in predict_text: return '2'
        if 'CC' in predict_text: return '3'
        if 'CB' in predict_text: return '4'
        if 'BB' in predict_text: return '5'
        if 'BA' in predict_text: return '6'
        if 'AA' in predict_text: return '7'
        return 'None'


    def init_task(self,task_description):
        self._interaction(task_description)

    def predict_each_student(self,persona):
        prediction_answer = self._interaction(persona)
        prediction_output = self._extract_prediction(prediction_answer)
        return prediction_output


def concatenate_result(raw_dataset_path,predict_result_path,output_path):
    raw_dataset_table = pd.read_csv(raw_dataset_path)
    raw_dataset_arr = np.array(raw_dataset_table)

    new_header = list(raw_dataset_table.columns.values)+['predict','run_id']

    predict_result_table = pd.read_csv(predict_result_path)
    
    final_dataset_arr = np.empty((0,len(new_header)))
    for i,raw_item in enumerate(raw_dataset_arr):
        student_id = raw_item[0]
        predict_result_items = predict_result_table[predict_result_table['sample_id']==student_id]

        run_id_arr = np.array(predict_result_items['run_id'])
        run_id_arr = run_id_arr.reshape((len(run_id_arr),1))
        predict_arr = np.array(predict_result_items['predict'])
        predict_arr = predict_arr.reshape((len(predict_arr),1))
        predict_run_arr = np.concatenate((predict_arr,run_id_arr),axis=1)

        raw_item_arr = np.array(raw_item)
        raw_item_arr = raw_item_arr.reshape((1,len(raw_item_arr)))
        student_info_arr = np.tile(raw_item_arr,(len(predict_run_arr),1))
        student_item_arr = np.concatenate((student_info_arr,predict_run_arr),axis=1)
        final_dataset_arr = np.concatenate((final_dataset_arr,student_item_arr),axis=0)

    final_dataset = pd.DataFrame(final_dataset_arr,columns=new_header)
    final_dataset.to_csv(output_path,index=False)
