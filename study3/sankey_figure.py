import plotly.graph_objects as go

exp_list = ['Experiment 1','Experiment 2','Experiment 3-Task 1','Experiment 3-Task 2']
exp_2_type_list = ['Type i','Type ii','Type iii','Type iv','Type v','Type vi','Type vii','Type viii']
exp_3_type_list = ['Type a','Type b','Type c','Type 1a','Type 1b','Type 1c','Type 2a','Type 2b','Type 2c','Type 3a','Type 3b','Type 3c']
llm_input_list = ['Demographics','All Past Assessments','Past One Assessment','Past Two Assessments','Past Three Assessments','Past Four Assessments','Past Five Assessments',
                  'All Slides','Individual Slide','Selected Slide: Past Understanding','Post-Test Questions','All Past Understanding',
                  'Average Score: Pre-Test Questions','Slides Related to Current Question','Related Slides: Understanding','Engagement in the Course']
llm_output_list = ['Final Grade','Final Exam Score','Individual Slide: Understanding','Post-Test: Average Accuracy','Post-Test: Each Question Accuracy','Post-Test: Specific Question Accuracy']
extra = ['Current Post-Test Question']

# exp_list = ['0-Experiment 1','1-Experiment 2','2-Experiment 3-Task 1','3-Experiment 3-Task 2']
# exp_2_type_list = ['4-Type i','5-Type ii','6-Type iii','7-Type iv','8-Type v','9-Type vi','10-Type vii','11-Type viii']
# exp_3_type_list = ['12-Type a','13-Type b','14-Type c','15-Type 1a','16-Type 1b','17-Type 1c','18-Type 2a','19-Type 2b','20-Type 2c','21-Type 3a','22-Type 3b','23-Type 3c']
# llm_input_list = ['24-Demographics','25-All Past Assessments','26-Past One Assessment','27-Past Two Assessments','28-Past Three Assessments','29-Past Four Assessments','30-Past Five Assessments',
#                   '31-All Slides','32-Individual Slide','33-Selected Slide: Past Understanding','34-Post-Test Questions','35-All Past Understanding',
#                   '36-Average Score: Pre-Test Questions','37-Slides Related to Current Question','38-Related Slides: Understanding','39-Engagement in the Course']
# llm_output_list = ['40-Final Grade','41-Final Exam Score','42-Individual Slide: Understanding','43-Post-Test: Average Accuracy','44-Post-Test: Each Question Accuracy','45-Post-Test: Specific Question Accuracy']
# extra = ['46-Current Post-Test Question']

source_list = [0,24,1,4,24,1,5,25,1,6,6,24,25,1,1,1,1,1,7,8,9,10,11,26,27,28,29,30,2,2,2,12,12,24,31,13,13,24,32,14,14,14,24,32,33,3,3,3,3,3,3,3,3,3,15,15,15,24,31,34,16,16,16,16,24,31,35,34,17,17,17,17,17,24,31,35,34,36,18,18,18,24,31,34,19,19,19,19,24,31,35,34,20,20,20,20,20,24,31,35,34,36,21,21,21,21,24,37,38,46,22,22,22,22,22,24,37,38,36,46,23,23,23,23,23,23,24,37,38,36,39,46]
target_list = [24,40,4,24,41,5,25,41,6,24,25,41,41,7,8,9,10,11,26,27,28,29,30,41,41,41,41,41,12,13,14,24,31,42,42,24,32,42,42,24,32,33,42,42,42,15,16,17,18,19,20,21,22,23,24,31,34,43,43,43,24,31,35,34,43,43,43,43,24,31,35,34,36,43,43,43,43,43,24,31,34,44,44,44,24,31,35,34,44,44,44,44,24,31,35,34,36,44,44,44,44,44,24,37,38,46,45,45,45,45,24,37,38,36,46,45,45,45,45,45,24,37,38,36,39,46,45,45,45,45,45,45]
all_node_list = exp_list + exp_2_type_list + exp_3_type_list + llm_input_list + llm_output_list + extra

color_dict = ['green']
color_list = []
alpha = 0.5
for s,source_item in enumerate(source_list):
  target_item = target_list[s]
  if source_item in [0] or target_item in [40]:
    color_list.append("rgba(227, 103, 50, "+str(alpha)+")")
  elif source_item in [1,4,5,6,7,8,9,10,11] or target_item in [41]:
    color_list.append("rgba(29, 224, 81, "+str(alpha)+")")
  elif source_item in [2,12,13,14] or target_item in [42]:
    color_list.append("rgba(44, 175, 222, "+str(alpha)+")") 
  elif source_item in [15,16,17] or target_item in [15,16,17,43]:
    color_list.append("rgba(154, 166, 160, "+str(alpha)+")")
  elif source_item in [18,19,20] or target_item in [18,19,20,44]:
    color_list.append("rgba(191, 27, 131, "+str(alpha)+")")
  elif source_item in [21,22,23] or target_item in [21,22,23,45]:
    color_list.append("rgba(176, 53, 53, "+str(alpha)+")")

color_node_list = []
alpha2 = 0.6
for s,source_item in enumerate(all_node_list):
  if source_item in ['Experiment 1','Final Grade']:
    color_node_list.append("rgba(227, 103, 50, "+str(alpha2)+")")
  elif source_item in ['Experiment 2','Final Exam Score','Type i','Type ii','Type iii','Type iv','Type v','Type vi','Type vii','Type viii','All Past Assessments','Past One Assessment','Past Two Assessments','Past Three Assessments','Past Four Assessments','Past Five Assessments']:
    color_node_list.append("rgba(29, 224, 81, "+str(alpha2)+")")
  elif source_item in ['Experiment 3-Task 1','Demographics','Type a','Type b','Type c','Individual Slide: Understanding','Individual Slide','Selected Slide: Past Understanding']:
    color_node_list.append("rgba(44, 175, 222, "+str(alpha2)+")") 
  elif source_item in ['Type 1a','Type 1b','Type 1c','Post-Test: Average Accuracy','All Slides']:
    color_node_list.append("rgba(154, 166, 160, "+str(alpha2)+")")
  elif source_item in ['Type 2a','Type 2b','Type 2c','Post-Test: Each Question Accuracy','Post-Test Questions','All Past Understanding','Average Score: Pre-Test Questions']:
    color_node_list.append("rgba(191, 27, 131, "+str(alpha2)+")")
  elif source_item in ['Type 3a','Type 3b','Type 3c','Post-Test: Specific Question Accuracy','Slides Related to Current Question','Related Slides: Understanding','Engagement in the Course','Current Post-Test Question']:
    color_node_list.append("rgba(176, 53, 53, "+str(alpha2)+")")
  elif source_item in ['Experiment 3-Task 2']:
    color_node_list.append("rgba(179, 4, 4, "+str(alpha2)+")")
  else:
    color_node_list.append("rgba(245, 66, 66, "+str(alpha2)+")")


fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 20,
      thickness = 50,
      line = dict(color = "black", width = 0.5),
      label = all_node_list,
      x = [0.01,0.01,0.01,0.01]+[0.2 for t in range(20)]+[0.5 for t in range(len(llm_input_list))]+[0.99 for t in range(len(llm_output_list))]+[0.5],
      y = [0.01,0.12,0.3,0.7]+[0.01+0.03*q for q in range(8)]+[0.25+0.05*q for q in range(3)]+[0.40+0.07*q for q in range(9)]+[0.03]+[0.15+0.035*q for q in range(6)]+[0.40,0.47,0.51,0.57,0.64,0.71,0.78,0.85,0.92]+[0.01,0.15,0.3,0.46,0.67,0.89]+[0.96],
      color = color_node_list
    ),
    link = dict(
      source = source_list, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target_list,
      # value = [1 for i in range(len(source_list))],
      value = [1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,4,5,3,4,5,4,5,6,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #   label = ["A1B1", "A2B2", "B1", "B2", "C1", "C2"],
      color = color_list,
  ))])

fig.update_layout(font_size=30)
# fig.write_html('plot.html', auto_open=True) 

fig.write_image('plot.pdf',width=2200, height=1200)