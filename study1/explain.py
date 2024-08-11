import pandas as pd
import tiktoken
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 
import seaborn as sns
import pacmap,umap
import openai
import os,sys,uuid,time,math 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from utils import generate_persona,concatenate_embedding_dataset,dataset_config_label


from openai import OpenAI
client = OpenAI(api_key = '')

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embedding_table(data_file,target_column,output_file,embedding_model="text-embedding-3-small",embedding_encoding="cl100k_base",max_tokens=8000):
    # the maximum max_tokens for text-embedding-3-small is 8191
    t1 = time.time()
    data_table = pd.read_csv(data_file,sep='\t')
    encoding = tiktoken.get_encoding(embedding_encoding)
    data_table[target_column] = data_table[target_column].astype(str)
    data_table["n_tokens"] = data_table[target_column].apply(lambda x: len(encoding.encode(x)))
    data_table_outlier = data_table[data_table.n_tokens > max_tokens]
    if len(data_table_outlier) != 0:
        raise ValueError(f'There exists tokens larger than max token: \n{data_table_outlier}')
    # data_table = data_table[data_table.n_tokens <= max_tokens].tail(top_n)
    data_table["embedding"] = data_table[target_column].apply(lambda x: get_embedding(x, model=embedding_model))
    data_table.to_csv(output_file,sep='\t',index=False)
    t2 = time.time()
    print('embedding generation time: ',t2-t1)


def rgb(hex):
    hex = hex[1:]
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal/256)
    rgb.append(1)
    return tuple(rgb)

def plot_region(r):
    plt.hlines(y = r[0][0], xmin=r[1][0], xmax=r[1][1], linewidth=1.5, linestyle="dotted", color="black")
    plt.hlines(y = r[0][1], xmin=r[1][0], xmax=r[1][1], linewidth=1.5, linestyle="dotted", color="black")
    plt.vlines(x = r[1][0], ymin=r[0][0], ymax=r[0][1], linewidth=1.5, linestyle="dotted", color="black")
    plt.vlines(x = r[1][1], ymin=r[0][0], ymax=r[0][1], linewidth=1.5, linestyle="dotted", color="black")
    
def plot_square(s, width):
    sx = s[0]
    sy = s[1]
    plt.hlines(y = sy, xmin=sx, xmax=sx+width, linewidth=2, linestyle="dotted", color="black")
    
    plt.hlines(y = sy - width, xmin=sx, xmax=sx + width, linewidth=2, linestyle="dotted", color="black")
    plt.vlines(x = sx, ymin=sy-width, ymax=sy, linewidth=2, linestyle="dotted", color="black")
    plt.vlines(x = sx + width, ymin=sy-width, ymax=sy, linewidth=2, linestyle="dotted", color="black")

def insert_subplot(x_coords, y_coords, label_coords, detail_coords, ax, region, box, c, loc1, loc2, s=200, linewidths=0.2, alpha=1, offset=0.1):
    # Create inset axes for region A.
    ax_inset = ax.inset_axes(box)  # These are figure coordinates.
    # if loc1 == None or loc2 == None:
    #     ax.indicate_inset_zoom(ax_inset, edgecolor="black", linestyle= (0,(0,0,1,1)), linewidth=1.5)
    # else:
    mark_inset(ax, ax_inset, loc1=loc1, loc2=loc2)
    
    ax_inset.scatter(x_coords, y_coords, c=c,  s=s, edgecolors="white", linewidths=linewidths, alpha=alpha)
    # ax_inset.set_facecolor('whitesmoke')
    ax_inset.set_xlim(region[0], region[1])
    ax_inset.set_ylim(region[2], region[3])
    # Connector Settings
    
    # Labels and Axis
    for spine in ax_inset.spines.values():
        spine.set_linestyle((0,(0,0,2,2.5))) 
        spine.set_visible(True)
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    
    ax_inset.tick_params(axis='both', which='both', length=0)
    
    # Annotations    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if (region[0]+ offset) <= x<= (region[1] - offset) and (region[2] + offset) <= y <= (region[3] - offset):
            label = label_coords[i]  # This is your annotation text, replace it with your desired text
            detail = detail_coords[i]  # This is your annotation text, replace it with your desired text
            ax_inset.set_xlabel(label)
            ax_inset.annotate(process_text(dataset_config_label[label][detail]),  # This is the text to use for the annotation
                    (x, y),  # This sets the location of the point to annotate
                    textcoords="offset points",  # how to position the text
                    xytext=(0.4,0.4),  # distance from text to points (x,y)
                    ha='center',  # horizontal alignment can be left, right or center
                    fontsize=10)  # font size

def process_text(x):
    return f''+x.replace("_", "-")


def visual_space_embedding(data_file,embedding_column,class_column,vis_model):
    from scipy.cluster.vq import whiten
    import cmcrameri.cm as cmc

    data_table = pd.read_csv(data_file,sep='\t')
    data_table = data_table[(data_table['demo_type']!='course_id')&(data_table['demo_type']!='cuml_gpa')&(data_table['demo_type']!='exp_gpa')]
    class_label = data_table[class_column].values
    # data_table['demo_content'] = data_table['demo_content'].astype(str)
    class_label_content = data_table['demo_choice_id'].values
    matrix = data_table[embedding_column].apply(eval).to_list()
    matrix = np.array(matrix)
    print('matrix.shape:',matrix.shape)
    class_all_list = list(set(data_table[class_column]))
    print('class_all_list: ',class_all_list)

    xh = whiten(matrix)

    projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, random_state=0, MN_ratio=1, FP_ratio=10, distance="angular", lr=0.5)
    xp = projector.fit_transform(xh)

    x_coords = xp[:,0]  
    y_coords = xp[:,1] 

    fig, ax = plt.subplots(figsize=(10,10))

    colors = list(mcolors.CSS4_COLORS.values())
    # colors = list(mcolors.TABLEAU_COLORS.values())
    if len(class_all_list) > len(colors):
        extra_colors = list(mcolors.CSS4_COLORS.values())
        colors.extend(extra_colors[:len(class_all_list) - len(colors)])
    color_mapping = {value: colors[i] for i, value in enumerate(class_all_list)}

    c = []
    cmap = cmc.batlowS
    # cmap = cmc.batlowS
    for l in class_label:
        c.append(color_mapping[l])
        
    # Plot the main scatter plot.
    # ax.set_facecolor('lightgrey')
    ax.scatter(xp[:,0], xp[:,1], c=c,  s=100, edgecolors="white", linewidths=0.5, alpha=0.8)
    
    # ax.set_xlim(-50,50)
    # ax.set_ylim(-50,50)

    ### Region S
    
    # region [x1,x2,y1,y2]
    # box [x0, y0, width, height]
    region_grade = [-9, 1, 8, 15]
    box_1 = [-0.6, 0.6, 0.6, 0.6]

    region_study_hour = [-19.5, -15, -9, -5.8]
    box_3 = [-0.6, 0, 0.6, 0.55]

    region_sibling = [7.8, 11, 6, 9]
    box_6 = [1, 0.65, 0.4, 0.55]

    region_transport = [-2.5, 3.5, 0, 4.7]
    box_9 = [0.425, 0, 0.3, 0.4]

    region_parent = [8, 16, -11.5, -4.2]
    box_15 = [1, 0, 0.4, 0.61]
    
    insert_subplot(x_coords, y_coords, class_label, class_label_content, ax, region_grade, box_1, c, loc1 = 1, loc2 = 4)
    insert_subplot(x_coords, y_coords, class_label, class_label_content, ax, region_study_hour, box_3, c, loc1 = 1, loc2 = 4)
    insert_subplot(x_coords, y_coords, class_label, class_label_content, ax, region_sibling, box_6, c, loc1 = 2, loc2 = 3)
    insert_subplot(x_coords, y_coords, class_label, class_label_content, ax, region_transport, box_9, c, loc1 = 1, loc2 = 2)
    insert_subplot(x_coords, y_coords, class_label, class_label_content, ax, region_parent, box_15, c, loc1 = 2, loc2 = 3)


    plt.autoscale()
    plt.axis("off")
    # Show the plot.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig("temp.pdf", format="pdf", bbox_inches="tight")



# Reference: 
# part of the codes come from paper: Germans Savcisens et al. Using sequences of life-events to predict human lives, Nature Computational Science 2023


persona_space_file = 'persona_space.csv'
embedding_space_file = 'persona_space_embedding.csv'

get_embedding_table(persona_space_file,'demo_content',embedding_space_file,embedding_model="text-embedding-3-small",embedding_encoding="cl100k_base",max_tokens=8000)

visual_space_embedding(embedding_space_file,'embedding','demo_type','tsne')
