# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 03:14:41 2023

@author: Freddy J. Orozco R.
@Powered: WinStats.
"""


import streamlit as st
import hydralit_components as hc
import datetime
import base64
import pandas as pd
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mplt
import matplotlib.font_manager as font_manager
import mplsoccer
from mplsoccer import Pitch, VerticalPitch, FontManager
import sklearn
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter
import seaborn as sns
from matplotlib import colors as mcolors
import requests
from PIL import Image
from matplotlib.patches import Rectangle
import math
import ScraperFC as sfc

############################################################################################################################################################################################################################
def split_minute(minute_str):
    if '+' in minute_str:
        parts = minute_str.split('+')
        return parts[0], parts[1]
    else:
        return minute_str, ''

def calcular_radio_desde_gol(x, y, y_gol):
    radio = math.sqrt(x**2 + (y - y_gol)**2)
    return radio
def es_deep_completion(row):
    radio = math.sqrt(row['X1']**2 + (row['Y1'] - y_gol_oponente)**2)
    return radio <= radio_umbral
def esta_dentro_semicircunferencia(x, y):
    distancia_centro = math.sqrt((x - punto_medio_cancha_rival[0])**2 + (y - punto_medio_cancha_rival[1])**2)
    return distancia_centro <= radio_umbral

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_entre_puntos(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def colorlist(color1, color2, num):
    """Generate list of num colors blending from color1 to color2"""
    result = [np.array(color1), np.array(color2)]
    while len(result) < num:
        temp = [result[0]]
        for i in range(len(result)-1):
            temp.append(np.sqrt((result[i]**2+result[i+1]**2)/2))
            temp.append(result[i+1])
        result = temp
    indices = np.linspace(0, len(result)-1, num).round().astype(int)
    return [result[i] for i in indices] 

hex_list2 = ['#121214', '#D81149', '#FF0050']
#hex_list = ['#121214', '#112F66', '#004DDD']B91845
hex_list4 = ['#5A9212', '#70BD0C', '#83E604']
#hex_list1 = ['#121214', '#854600', '#C36700']
hex_list = ['#121214', '#545454', '#9F9F9F']
hex_list1 = ['#121214', '#695E00', '#C7B200']
#hex_list2 = ['#121214', '#112F66', '#004DDD']
#hex_list = ['#121214', '#11834C', '#00D570']
cmap = sns.cubehelix_palette(start=.25, rot=-.3, light=1, reverse=True, as_cmap=True)
cmap2 = sns.diverging_palette(250, 344, as_cmap=True, center="dark")
cmap3 = sns.color_palette("dark:#FF0046", as_cmap=True)


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

#####################################################################################################################################################

font_path = 'Resources/keymer-bold.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop2 = font_manager.FontProperties(fname=font_path)

font_path2 = 'Resources/BasierCircle-Italic.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path2)
prop3 = font_manager.FontProperties(fname=font_path2)

#####################################################################################################################################################


###########################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################

#make it look nice from the start
st.set_page_config(layout='wide')

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.app.goo.gl/LFCobouKtT7oZ7Qv7")
    }
   .sidebar .sidebar-content {
        background: url("https://images.app.goo.gl/LFCobouKtT7oZ7Qv7")
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
# specify the primary menu definition
menu_data = [
    {'id': "ExploreTeamData", 'label':"Explore Data"},
    {'id': "DataScraping", 'label':"DataScraping"}
    #{'id': "Dashboard", 'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"} #can add a tooltip message]
]
over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    #login_name='Logout',
    hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################

if menu_id == "ExploreTeamData":
    with st.sidebar:
        with open("Resources/win.png", "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
            st.sidebar.markdown(
                f"""
                <div style="display:table;margin-top:-20%">
                    <img src="data:image/png;base64,{data}" width="300">
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("""---""")
        with st.form(key='form4'):
            uploaded_file = st.file_uploader("Choose a excel file", type="xlsx")

            DataMode = st.checkbox("Activate calculated columns")
            submit_button2 = st.form_submit_button(label='Aceptar')
    st.title("EXPLORE DATA")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

    dfORIG = df
    st.dataframe(df)
    st.divider()
    menuoptexpdata01, menuoptexpdata02, menuoptexpdata03 = st.columns(3)
    with menuoptexpdata01:
        TeamsOption = df['Team'].drop_duplicates().tolist()
        TeamSelExpData = st.selectbox("Seleccionar Equipo:", TeamsOption)
        df = df[df['Team'] == TeamSelExpData].reset_index(drop=True)
    with menuoptexpdata02:
        PlotVizOptionsd = ['Acciones', 'Pases', 'Remates']
        PlotVizSelExpData = st.selectbox("Seleccionar Gráfico:", PlotVizOptionsd)
    with menuoptexpdata03:
        PlotVizOption = ['Acciones', 'Pases', 'Remates', 'Recuperaciones']
        PlotVizSelExpData = st.selectbox("Seleccionar Gráfico:", PlotVizOption)

    st.divider()
    MaxAddMin = df['EfectiveMinute'].max()
    if PlotVizSelExpData == "Acciones":
        pltmnop01, pltmnop02, pltmnop03 = st.columns(3)
        with pltmnop01:
            OptionPlot = ['Touches Map', 'Touches Opponent Field', 'Territory Actions', 'Heatmap - Opponent Field', 'Heatmap - Zones', 'Heatmap - Gaussian', 'Heatmap - Kernel', 'Field Tilt']
            OptionPlotSel = st.selectbox('Seleccionar tipo gráfico:', OptionPlot)
        with pltmnop02:
            EfectMinSel = st.slider('Seleccionar rango de partido:', 0, MaxAddMin, (0, MaxAddMin))
        if OptionPlotSel == 'Territory Actions': 
            with pltmnop03:
                ColorOptionSel = st.color_picker('Selecciona color:', '#FF0046')
                colorviz = ColorOptionSel
        else:
            with pltmnop03:
                SelOpt = ['WinStats', 'FD']
                ColorOptionSel = st.selectbox('Selecciona color:', SelOpt)
        pltmain01, pltmain02 = st.columns(2)
        with pltmain01:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            if (OptionPlotSel == 'Touches Opponent Field') | (OptionPlotSel == 'Heatmap - Opponent Field'):
                pitch = VerticalPitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1.0, goal_type='box', pitch_length=105, pitch_width=68)

            else:
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1.0, goal_type='box', pitch_length=105, pitch_width=68)
                #Adding directon arrow
                ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
                ax29.axis("off")
                ax29.set_xlim(0,10)
                ax29.set_ylim(0,10)
                ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
                #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
                ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            pitch.draw(ax=ax)
            
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            dfKK = df

            if ColorOptionSel == 'WinStats':
                hex_list2 = ['#121214', '#D81149', '#FF0050']
                hex_list = ['#121214', '#545454', '#9F9F9F']
                colorviz = "#FF0050"
                # Definir los colores base con transparencias diferentes
                red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0]   # 121214
                green = [0.6, 0.1098039215686275, 0.2431372549019608, 0.6]   # 991C3E
                blue = [1, 0, 0.2745098039215686, 0.8]   # FF0046
                # Crear una lista de los colores y las posiciones en el colormap
                colors = [red, green, blue]
                positions = [0, 0.5, 1]
                # Crear el colormap continuo con transparencias
                cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
            if ColorOptionSel == 'FD':
                hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
                hex_list = ['#121214', '#545454', '#9F9F9F']
                colorviz = "#83E604"
                # Definir los colores base con transparencias diferentes
                red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0.2]   # 121214
                green = [0.3215686274509804, 0.5215686274509804, 0.0666666666666667, 0.5]   # 0059FF
                blue = [0.5137254901960784, 0.9019607843137255, 0.0156862745098039, 0.70]   # 3A7FFF
                # Crear una lista de los colores y las posiciones en el colormap
                colors = [red, green, blue]
                positions = [0, 0.5, 1]
                # Crear el colormap continuo con transparencias
                cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
            #df = dfKK.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
            #st.write(df)
            if OptionPlotSel == 'Touches Map': 
                
                #df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df

                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, -0.5, 'ACCIONES \nREALIZADAS', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                ax9.scatter(8, 5, s=320, color=colorviz, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                ax9.text(8, -0.5, 'TERRITORIO\nRECURRENTE', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            if OptionPlotSel == 'Touches Opponent Field':
                
                #df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
                df.rename(columns={'X1':'Y1', 'Y1':'X1'}, inplace=True)
                df = df[df['Y1'] >= 52.5].reset_index()

                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.set_ylim(52.3,110)
                #Adding title
                ax9 = fig.add_axes([0.16,0.135,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 1.0, 'ACCIONES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            if OptionPlotSel == 'Territory Actions': 
                
                #df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
                
                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
                scaler  = StandardScaler()
                defpoints1 = df[['X1', 'Y1']].values
                defpoints2 = scaler.fit_transform(defpoints1)
                df2 = pd.DataFrame(defpoints2, columns = ['Xstd', 'Ystd'])
                df3 = pd.concat([df, df2], axis=1)
                df5=df3
                df3 = df3[df3['Xstd'] <= 1]
                df3 = df3[df3['Xstd'] >= -1]
                df3 = df3[df3['Ystd'] <= 1]
                df3 = df3[df3['Ystd'] >= -1].reset_index()
                df9 = df
                df = df3
                defpoints = df[['X1', 'Y1']].values
                #st.write(defpoints)

                hull = ConvexHull(df[['X1','Y1']])        
                ax.scatter(df9['X1'], df9['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                #Loop through each of the hull's simplices
                for simplex in hull.simplices:
                    #Draw a black line between each
                    ax.plot(defpoints[simplex, 0], defpoints[simplex, 1], '#BABABA', lw=2, zorder = 1, ls='--')
                ax.fill(defpoints[hull.vertices,0], defpoints[hull.vertices,1], colorviz, alpha=0.7)
                meanposx = df9['X1'].mean()
                meanposy = df9['Y1'].mean()
                ax.scatter(meanposx, meanposy, s=1000, color="w", edgecolors=colorviz, lw=2.5, zorder=25, alpha=0.95)
                #names = PlayerSelExpData.split()
                #iniciales = ""
                #for name in names:
                #   iniciales += name[0] 
                #names_iniciales = names_iniciales.squeeze().tolist()
                #ax.text(meanposx, meanposy, iniciales, color='k', fontproperties=prop2, fontsize=13, zorder=34, ha='center', va='center')
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, -0.5, 'ACCIONES \nREALIZADAS', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                ax9.scatter(8, 5, s=320, color=colorviz, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                ax9.text(8, -0.5, 'TERRITORIO\nRECURRENTE', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")

            elif OptionPlotSel == 'Heatmap - Opponent Field':
                
                #df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
                zone_areas = {
                    'zone_1':{
                        'x_lower_bound': 54.16, 'x_upper_bound': 68,
                        'y_lower_bound': 88.5, 'y_upper_bound': 105,
                    },
                    'zone_2':{
                        'x_lower_bound': 0, 'x_upper_bound': 13.84,
                        'y_lower_bound': 88.5, 'y_upper_bound': 105,
                    },
                    'zone_3':{
                        'x_lower_bound': 54.16, 'x_upper_bound': 68,
                        'y_lower_bound': 70.5, 'y_upper_bound': 88.5,
                    },
                    'zone_4':{
                        'x_lower_bound': 0, 'x_upper_bound': 13.84,
                        'y_lower_bound': 70.5, 'y_upper_bound': 88.5,
                    },
                    'zone_5':{
                        'x_lower_bound': 43.16, 'x_upper_bound': 54.16,
                        'y_lower_bound': 88.5, 'y_upper_bound': 105,
                    },
                    'zone_6':{
                        'x_lower_bound': 13.84, 'x_upper_bound': 24.84,
                        'y_lower_bound': 88.5, 'y_upper_bound': 105,
                    },
                    'zone_7':{
                        'x_lower_bound': 24.84, 'x_upper_bound': 43.16,
                        'y_lower_bound': 88.5, 'y_upper_bound': 105,
                    },
                    'zone_8':{
                        'x_lower_bound': 24.84, 'x_upper_bound': 43.16,
                        'y_lower_bound': 70.5, 'y_upper_bound': 88.5,
                    },
                    'zone_9':{
                        'x_lower_bound': 43.16, 'x_upper_bound': 54.16,
                        'y_lower_bound': 70.5, 'y_upper_bound': 88.5,
                    },
                    'zone_10':{
                        'x_lower_bound': 13.84, 'x_upper_bound': 24.84,
                        'y_lower_bound': 70.5, 'y_upper_bound': 88.5,
                    },
                    'zone_11':{
                        'x_lower_bound': 43.16, 'x_upper_bound': 54.16,
                        'y_lower_bound': 52.5, 'y_upper_bound': 70.5,
                    },
                    'zone_12':{
                        'x_lower_bound': 13.84, 'x_upper_bound': 24.84,
                        'y_lower_bound': 52.5, 'y_upper_bound': 70.5,
                    },
                    'zone_13':{
                        'x_lower_bound': 54.16, 'x_upper_bound': 68,
                        'y_lower_bound': 52.5, 'y_upper_bound': 70.5,
                    },
                    'zone_14':{
                        'x_lower_bound': 0, 'x_upper_bound': 13.84,
                        'y_lower_bound': 52.5, 'y_upper_bound': 70.5,
                    },
                    'zone_15':{
                        'x_lower_bound': 24.84, 'x_upper_bound': 43.16,
                        'y_lower_bound': 52.5, 'y_upper_bound': 70.5,
                    }
                }
                
                def assign_action_zone(x,y):
                    '''
                    This function returns the zone based on the x & y coordinates of the shot
                    taken.
                    Args:
                        - x (float): the x position of the shot based on a vertical grid.
                        - y (float): the y position of the shot based on a vertical grid.
                    '''
                
                    global zone_areas
                
                    # Conditions
                
                    for zone in zone_areas:
                        if (x >= zone_areas[zone]['x_lower_bound']) & (x <= zone_areas[zone]['x_upper_bound']):
                            if (y >= zone_areas[zone]['y_lower_bound']) & (y <= zone_areas[zone]['y_upper_bound']):
                                return zone
                
                
                zone_colors = {
                    'zone_1': 'black',
                    'zone_2': 'red',
                    'zone_3': 'blue',
                    'zone_4': 'yellow',
                    'zone_5': 'green',
                    'zone_6': 'pink',
                    'zone_7': 'purple',
                    'zone_8': 'grey',
                    'zone_9': 'brown',
                    'zone_10': 'lightblue',
                    'zone_11': 'lightcyan',
                    'zone_12': 'lightgrey',
                    'zone_13': 'w',
                    'zone_14': 'orange',
                    'zone_15': 'cyan'
                }
                
                df.rename(columns={'X1':'Y1', 'Y1':'X1'}, inplace=True)
                df = df[df['Y1'] >= 52.5].reset_index()

                df['zone_area'] = [assign_action_zone(x,y) for x,y in zip(df['X1'], df['Y1'])]




                data = df.groupby(['zone_area']).apply(lambda x: x.shape[0]).reset_index()
                data.rename(columns={0:'num_actions'}, inplace=True)
                data['pct_actions'] = data['num_actions']/df['index'].count()
                
                
                plot_df = data
                max_value = plot_df['pct_actions'].max()
                
                
                
                for zone in plot_df['zone_area']:
                    action_pct = plot_df[plot_df['zone_area'] == zone]['pct_actions'].iloc[0]
                    x_lim = [zone_areas[zone]['x_lower_bound'], zone_areas[zone]['x_upper_bound']]
                    y1 = zone_areas[zone]['y_lower_bound']
                    y2 = zone_areas[zone]['y_upper_bound']
                    ax.fill_between(
                        x=x_lim, 
                        y1=y1, y2=y2, 
                        color=colorviz, alpha=((action_pct/max_value)),
                        zorder=0, ec='None')
                    if action_pct > 0.005:
                        x_pos = x_lim[0] + abs(x_lim[0] - x_lim[1])/2
                        y_pos = y1 + abs(y1 - y2)/2
                        text_ = ax.annotate(
                            xy=(x_pos, y_pos),
                            text=f'{action_pct:.0%}',
                            ha='center',
                            va='center',
                            color='w',
                            fontproperties=prop2,
                            size=20
                        )
                        text_.set_path_effects(
                            [path_effects.Stroke(linewidth=1.0, foreground='k'), path_effects.Normal()]
                        )


                ax.plot([13.84, 13.84], [52.5, 105], ls='--', color='#9F9F9F')
                ax.plot([54.16, 54.16], [52.5, 105], ls='--', color='#9F9F9F')
                ax.plot([24.84, 24.84], [52.5, 105], ls='--', color='#9F9F9F')
                ax.plot([43.16, 43.16], [52.5, 105], ls='--', color='#9F9F9F')
                ax.plot([0, 68], [88.5, 88.5], ls='--', color='#9F9F9F')
                ax.plot([0, 68], [70.5, 70.5], ls='--', color='#9F9F9F')

                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
                ax.set_ylim(52.3,110)
                #Adding title
                ax9 = fig.add_axes([0.16,0.135,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, 1.0, 'ACCIONES', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
                
            elif OptionPlotSel == 'Heatmap - Zones':

                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
               
                path_eff = [path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]
                bin_statistic = pitch.bin_statistic_positional(df.X1, df.Y1, statistic='count', positional='full', normalize=True)
                pitch.heatmap_positional(bin_statistic, ax=ax, cmap=cmaps, edgecolors='#524F50', linewidth=1)
                pitch.scatter(df.X1, df.Y1, c='w', s=15, alpha=0.02, ax=ax)
                labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=14, fontproperties=prop2, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES REALIZADAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            elif OptionPlotSel == 'Heatmap - Gaussian':
                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
                
                bin_statistic = pitch.bin_statistic(df['X1'], df['Y1'], statistic='count', bins=(120, 80))
                bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 4)
                pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmaps, edgecolors=(0,0,0,0), zorder=-2)    
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES REALIZADAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            elif OptionPlotSel == 'Heatmap - Kernel':
                df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKKcleaned = df
                #bin_statistic = pitch.bin_statistic(df['X1'], df['Y1'], statistic='count', bins=(120, 80))
                #bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 4)
                kde = pitch.kdeplot(dfKKcleaned.X1, dfKKcleaned.Y1, ax=ax,
                    # fill using 100 levels so it looks smooth
                    fill=True, levels=500,
                    # shade the lowest area so it looks smooth
                    # so even if there are no events it gets some color
                    thresh=0,
                    cut=2, alpha=0.7, zorder=-2,  # extended the cut so it reaches the bottom edge
                    cmap=cmaps)

                
                #pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmaps, edgecolors=(0,0,0,0), zorder=-2)    
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES REALIZADAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
                
        with pltmain02:
            st.dataframe(dfKK[['ActionID', 'Event', 'Minute', 'EfectiveMinute',  'PlayerID', 'Player', 'Team', 'X1', 'Y1', 'X2', 'Y2']])
    if PlotVizSelExpData == "Remates":
        pltmnop01, pltmnop02, pltmnop03 = st.columns(3)
        with pltmnop01:
            OptionPlot = ['Shots Map', 'Shots Heatmap - Zones', 'Shots Heatmap - Gaussian']
            OptionPlotSel = st.selectbox('Seleccionar tipo gráfico:', OptionPlot)
        with pltmnop02:
            ListMatchs = df['MatchID'].drop_duplicates().tolist()
            ListMatchSel = st.multiselect('Seleccionar partidos:', ListMatchs)
            
        if OptionPlotSel == 'Shots Map': 
            with pltmnop03:
                ColorOptionSel = st.color_picker('Selecciona color:', '#FF0046')
                colorviz = ColorOptionSel
        else:
            with pltmnop03:
                SelOpt = ['WinStats', 'FD']
                ColorOptionSel = st.selectbox('Selecciona color:', SelOpt)
        pltmain01, pltmain02 = st.columns(2)
        with pltmain01:
            
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            #df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            
            df = df[(df['Event'] == 'Goals') | (df['Event'] == 'Shots on target') | (df['Event'] == 'Shots off target') | (df['Event'] == 'Blocks')].reset_index(drop=True)
            df = df[df['MatchID'].isin(ListMatchSel)]
            dfKK = df
            if ColorOptionSel == 'WinStats':
                hex_list2 = ['#121214', '#D81149', '#FF0050']
                hex_list = ['#121214', '#545454', '#9F9F9F']
                colorviz = "#FF0050"
                # Definir los colores base con transparencias diferentes
                red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0]   # 121214
                green = [0.6, 0.1098039215686275, 0.2431372549019608, 0.6]   # 991C3E
                blue = [1, 0, 0.2745098039215686, 0.8]   # FF0046
                # Crear una lista de los colores y las posiciones en el colormap
                colors = [red, green, blue]
                positions = [0, 0.5, 1]
                # Crear el colormap continuo con transparencias
                cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
            if ColorOptionSel == 'FD':
                hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
                hex_list = ['#121214', '#545454', '#9F9F9F']
                colorviz = "#83E604"
                # Definir los colores base con transparencias diferentes
                red = [0.0705882352941176, 0.0705882352941176, 0.0784313725490196, 0.2]   # 121214
                green = [0.3215686274509804, 0.5215686274509804, 0.0666666666666667, 0.5]   # 0059FF
                blue = [0.5137254901960784, 0.9019607843137255, 0.0156862745098039, 0.70]   # 3A7FFF
                # Crear una lista de los colores y las posiciones en el colormap
                colors = [red, green, blue]
                positions = [0, 0.5, 1]
                # Crear el colormap continuo con transparencias
                cmaps = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
            #df = dfKK.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
            #st.write(df)
            if OptionPlotSel == 'Shots Map': 
                
                #df = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #dfKKcleaned = df
                
                #df = df[df['Event'] != 'Assists'].reset_index(drop=True)
                dfKK = df
                  
                ax.scatter(df['X1'], df['Y1'], color = colorviz, edgecolors='w', s=30, zorder=2, alpha=0.2)
               
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKK)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                #Adding title
                ax9 = fig.add_axes([0.17,0.16,0.20,0.07])
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)
                ax9.scatter(2, 5, s=120, color=colorviz, edgecolors='#FFFFFF', lw=1)
                ax9.text(2, -0.5, 'ACCIONES \nREALIZADAS', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                ax9.scatter(8, 5, s=320, color=colorviz, edgecolors='#FFFFFF', lw=1, ls='--', marker='h')
                ax9.text(8, -0.5, 'TERRITORIO\nRECURRENTE', fontproperties=prop2, fontsize=9, ha='center', va='center', c='w')
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
        with pltmain02:
            st.dataframe(dfKK[['ActionID', 'Matchday', 'MatchID', 'Event', 'Minute', 'EfectiveMinute',  'PlayerID', 'Player', 'Team', 'X1', 'Y1', 'X2', 'Y2']])
    if PlotVizSelExpData == "Pases":
        pltmnop11, pltmnop12, pltmnop13 = st.columns(3)
        with pltmnop11:
            OptionPlot = ['Passes Viz', 'Progressive Passes', 'Passes Into Final Third', 'Passes Into Penalty Area', 'Long Passes', 'Passes Into Half Spaces']
            OptionPlotSel = st.selectbox('Seleccionar tipo gráfico:', OptionPlot)
        with pltmnop12:
            EfectMinSel = st.slider('Seleccionar rango de partido:', 0, MaxAddMin, (0, MaxAddMin))
        with pltmnop13:
                MetOption = ['WinStats', 'FD']
                MetOptionSel = st.selectbox('Choose color type:', MetOption)
        if MetOptionSel == 'WinStats':
            hex_list2 = ['#121214', '#D81149', '#FF0050']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#FF0050"
        if MetOptionSel == 'FD':
            hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#83E604"
        #if OptionPlot == 'Passes Map':
        #    with pltmnop13:
        #        MetOption = ['Pases Claves', 'Asistencias']
        #        MetOptionSel = st.selectbox('Seleccionar métrica:', MetOption)
        #if OptionPlot == 'Progressive Passes Map':
        #    with pltmnop13:
        #        ValorProg = st.select_slider('Valor progresivo:', 0, 10)
    
        
        pltmain11, pltmain12 = st.columns(2)
        with pltmain11:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=0.5, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            dfKK = df
            if OptionPlotSel == 'Passes Viz':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index()
                dfKKK = df
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                dfast = df[df['Event'] == 'Assists']
                dfkey = df[df['Event'] == 'Key Passes']
                dfpas = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')]
                
                #Progressive
                df['Beginning'] = np.sqrt(np.square(105-df['X1']) + np.square(34-df['Y1']))
                df['Ending']    = np.sqrt(np.square(105-df['X2']) + np.square(34-df['Y2']))
                df['Progress']  = [(df['Ending'][x]) / (df['Beginning'][x]) <= 0.8 for x in range(len(df.Beginning))]
                
                
                #Filter by passes progressives
                dfprog = df[df['Progress'] == True].reset_index()
                dfprog = dfprog.drop(['index'], axis=1)    
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                x1 = dfpas['X1']
                y1 = dfpas['Y1']
                x2 = dfpas['X2']
                y2 = dfpas['Y2']
                
                x1a = dfprog['X1']
                y1a = dfprog['Y1']
                x2a = dfprog['X2']
                y2a = dfprog['Y2']
                
                x1k = dfkey['X1']
                y1k = dfkey['Y1']
                x2k = dfkey['X2']
                y2k = dfkey['Y2']

                pitch.lines(x1, y1, x2, y2, cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True) 
                ax.scatter(x2, y2, color='#9F9F9F', edgecolors='#121214', zorder=3, lw=0.5)       
                    
                pitch.lines(x1a, y1a, x2a, y2a, cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(x2a, y2a, color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)           
                
                
                pitch.lines(x1k, y1k, x2k, y2k, cmap=get_continuous_cmap(hex_list1), ax=ax, lw=2, comet=True, transparent=True, zorder=10) 
                ax.scatter(x2k, y2k, color="#C7B200", edgecolors='#121214', zorder=5, lw=0.5)
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKK)) + " PASES COMPLETOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(52.5, 12, marker='s', color=colorviz, s=300)
                ax9.text(52.5, 2, 'PASES PROGRESIVOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(78.75, 12, marker='s', color='#C7B200', s=300)
                ax9.text(78.75, 2, 'PASES CLAVES', color='#C7B200', fontproperties=prop2, ha='center', fontsize=9)

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 

            if OptionPlotSel == 'Progressive Passes':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                #dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #Progressive
                df['Beginning'] = np.sqrt(np.square(105-df['X1']) + np.square(34-df['Y1']))
                df['Ending']    = np.sqrt(np.square(105-df['X2']) + np.square(34-df['Y2']))
                df['Progress']  = [(df['Ending'][x]) / (df['Beginning'][x]) <= 0.8 for x in range(len(df.Beginning))]
                                          
                #Filter by passes progressives
                dfprog = df[df['Progress'] == True].reset_index()
                dfprog = dfprog.drop(['index'], axis=1)
                dfprog = dfprog.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='first')
                dfw = dfprog[(dfprog['Event'] == 'Successful passes') | (dfprog['Event'] == 'Key Passes') | (dfprog['Event'] == 'Assists') | (dfprog['Event'] == 'Successful open play crosses') | (dfprog['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dff = dfprog[(dfprog['Event'] == 'Unsuccessful passes') | (dfprog['Event'] == 'Unsuccessful open play crosses') | (dfprog['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  

                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)     
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfprog)) + " PASES PROGRESIVOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                df = dfprog.reset_index(drop=True)
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 
            if OptionPlotSel == 'Passes Into Final Third':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                df = df[(df['X1'] <= 70) & (df['X2'] >= 70)].reset_index(drop=True)

                dfwin = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dffail = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                dfKKK = df
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                pitch.lines(dfwin['X1'], dfwin['Y1'], dfwin['X2'], dfwin['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True)
                ax.scatter(dfwin['X2'], dfwin['Y2'], color='#FF0050', edgecolors='#121214', zorder=3, lw=0.5)
                pitch.lines(dffail['X1'], dffail['Y1'], dffail['X2'], dffail['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True)
                ax.scatter(dffail['X2'], dffail['Y2'], color='#9F9F9F', edgecolors='#121214', zorder=3, lw=0.5)
                ax.vlines(x=70, ymin=0, ymax=68, color='w', alpha=0.3, ls='--', zorder=-1)
                ax.add_patch(Rectangle((70, 0), 35, 68, fc="#000000", fill=True, alpha=0.7, zorder=-2))

                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKK)) + " PASES HACIA ÚLTIMO TERCIO", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png") 
                #pitch.lines(x1a, y1a, x2a, y2a, cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                #ax.scatter(x2a, y2a, color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)
            if OptionPlotSel == 'Passes Into Penalty Area':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                # Coordenadas del cuadrilátero
                x1_cuadrilatero, y1_cuadrilatero = 88.5, 13.84
                x2_cuadrilatero, y2_cuadrilatero = 105, 13.84
                x3_cuadrilatero, y3_cuadrilatero = 88.5, 54.16
                x4_cuadrilatero, y4_cuadrilatero = 105, 54.16
                
                # Primera condición: X1, Y1 deben estar por fuera del cuadrilátero
                condicion1 = (
                    (df['X1'] < x1_cuadrilatero) |    # X1 debe ser menor que x1_cuadrilatero
                    (df['Y1'] < y1_cuadrilatero) |    # Y1 debe ser menor que y1_cuadrilatero
                    (df['X1'] > x4_cuadrilatero) |    # X1 debe ser mayor que x4_cuadrilatero
                    (df['Y1'] > y3_cuadrilatero)      # Y1 debe ser mayor que y3_cuadrilatero
                )
                
                # Segunda condición: X2, Y2 deben estar por dentro del cuadrilátero
                condicion2 = (
                    (df['X2'] >= x1_cuadrilatero) &   # X2 debe ser mayor o igual que x1_cuadrilatero
                    (df['Y2'] >= y1_cuadrilatero) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
                    (df['X2'] <= x4_cuadrilatero) &   # X2 debe ser menor o igual que x4_cuadrilatero
                    (df['Y2'] <= y3_cuadrilatero)     # Y2 debe ser menor o igual que y3_cuadrilatero
                )
                
                # Aplicar las condiciones para filtrar el DataFrame
                df = df[condicion1 & condicion2]
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)

                dfw = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dff = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  

                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)  
                ax.vlines(x=88.5, ymin=13.84, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.vlines(x=105, ymin=13.84, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=88.5, xmax=105, y=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=88.5, xmax=105, y=13.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.add_patch(Rectangle((88.5, 13.84), 16.5, 40.32, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKK)) + " PASES HACIA ÁREA RIVAL", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
                
            if OptionPlotSel == "Passes Into Half Spaces":
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                #df = df[(df['X1'] >= 52.5) & (df['X1'] <= 88.5)].reset_index(drop=True)
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)

                # Coordenadas del cuadrilátero A
                x1_cuadrilatero_A, y1_cuadrilatero_A = 52.5, 43.16
                x2_cuadrilatero_A, y2_cuadrilatero_A = 88.5, 43.16
                x3_cuadrilatero_A, y3_cuadrilatero_A = 52.5, 54.16
                x4_cuadrilatero_A, y4_cuadrilatero_A = 88.5, 54.16
                
                # Primera condición: X1, Y1 deben estar por fuera del cuadrilátero
                condicion1_A = (
                    (df['X1'] < x1_cuadrilatero_A) |    # X1 debe ser menor que x1_cuadrilatero
                    (df['Y1'] < y1_cuadrilatero_A) |    # Y1 debe ser menor que y1_cuadrilatero
                    (df['X1'] > x4_cuadrilatero_A) |    # X1 debe ser mayor que x4_cuadrilatero
                    (df['Y1'] > y3_cuadrilatero_A)      # Y1 debe ser mayor que y3_cuadrilatero
                )
                
                # Segunda condición: X2, Y2 deben estar por dentro del cuadrilátero
                condicion2_A = (
                    (df['X2'] >= x1_cuadrilatero_A) &   # X2 debe ser mayor o igual que x1_cuadrilatero
                    (df['Y2'] >= y1_cuadrilatero_A) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
                    (df['X2'] <= x4_cuadrilatero_A) &   # X2 debe ser menor o igual que x4_cuadrilatero
                    (df['Y2'] <= y3_cuadrilatero_A)     # Y2 debe ser menor o igual que y3_cuadrilatero
                )

                # Coordenadas del cuadrilátero B
                x1_cuadrilatero_B, y1_cuadrilatero_B = 52.5, 13.84
                x2_cuadrilatero_B, y2_cuadrilatero_B = 88.5, 13.84
                x3_cuadrilatero_B, y3_cuadrilatero_B = 52.5, 24.84
                x4_cuadrilatero_B, y4_cuadrilatero_B = 88.5, 24.84
                
                # Primera condición: X1, Y1 deben estar por fuera del cuadrilátero
                condicion1_B = (
                    (df['X1'] < x1_cuadrilatero_B) |    # X1 debe ser menor que x1_cuadrilatero
                    (df['Y1'] < y1_cuadrilatero_B) |    # Y1 debe ser menor que y1_cuadrilatero
                    (df['X1'] > x4_cuadrilatero_B) |    # X1 debe ser mayor que x4_cuadrilatero
                    (df['Y1'] > y3_cuadrilatero_B)      # Y1 debe ser mayor que y3_cuadrilatero
                )
                
                # Segunda condición: X2, Y2 deben estar por dentro del cuadrilátero
                condicion2_B = (
                    (df['X2'] >= x1_cuadrilatero_B) &   # X2 debe ser mayor o igual que x1_cuadrilatero
                    (df['Y2'] >= y1_cuadrilatero_B) &   # Y2 debe ser mayor o igual que y1_cuadrilatero
                    (df['X2'] <= x4_cuadrilatero_B) &   # X2 debe ser menor o igual que x4_cuadrilatero
                    (df['Y2'] <= y3_cuadrilatero_B)     # Y2 debe ser menor o igual que y3_cuadrilatero
                )
                
                # Aplicar las condiciones para filtrar el DataFrame
                df = df[(condicion1_A & condicion2_A) | (condicion1_B & condicion2_B)].reset_index(drop=True)

                dfw = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dff = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  
                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)
                
                ax.vlines(x=52.5, ymin=43.16, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.vlines(x=88.5, ymin=43.16, ymax=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=52.5, xmax=88.5, y=54.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=52.5, xmax=88.5, y=43.16, color='w', alpha=1, ls='--', lw=2, zorder=-1)

                ax.vlines(x=52.5, ymin=13.84, ymax=24.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.vlines(x=88.5, ymin=13.84, ymax=24.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=52.5, xmax=88.5, y=24.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                ax.hlines(xmin=52.5, xmax=88.5, y=13.84, color='w', alpha=1, ls='--', lw=2, zorder=-1)
                #ax.add_patch(Rectangle((88.5, 13.84), 16.5, 40.32, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.add_patch(Rectangle((52.5, 13.84), 36, 11, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                ax.add_patch(Rectangle((52.5, 43.16), 36, 11, fc="#000000", fill=True, alpha=0.7, zorder=-2))
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(df)) + " PASES HACIA ESPACIOS INTERMEDIOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            if OptionPlotSel == 'Long Passes':
                df = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses') | (df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index()
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
                df['Distance'] = np.sqrt((df['X2'] - df['X1'])**2 + (df['Y2'] - df['Y1'])**2)
                df = df[df['Distance'] >= 32].reset_index(drop=True)
                #df = df.drop(['Distance'], axis=1)

                
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
        

                dfw = df[(df['Event'] == 'Successful passes') | (df['Event'] == 'Key Passes') | (df['Event'] == 'Assists') | (df['Event'] == 'Successful open play crosses') | (df['Event'] == 'Successful set play crosses')].reset_index(drop=True)
                dff = df[(df['Event'] == 'Unsuccessful passes') | (df['Event'] == 'Unsuccessful open play crosses') | (df['Event'] == 'Unsuccessful set play crosses')].reset_index(drop=True)
                
                pitch.lines(dfw['X1'], dfw['Y1'], dfw['X2'], dfw['Y2'], cmap=get_continuous_cmap(hex_list2), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dfw['X2'], dfw['Y2'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5)  
                pitch.lines(dff['X1'], dff['Y1'], dff['X2'], dff['Y2'], cmap=get_continuous_cmap(hex_list), ax=ax, lw=2, comet=True, transparent=True, zorder=3) 
                ax.scatter(dff['X2'], dff['Y2'], color="#9F9F9F", edgecolors='#121214', zorder=3, lw=0.5)

                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(df)) + " PASES LARGOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(32.5, 15, marker='s', color=colorviz, s=300)
                ax9.text(32.5, 0, 'PASES\nEXITOSOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                ax9.scatter(72.5, 15, marker='s', color='#9F9F9F', s=300)
                ax9.text(72.5, 0, 'PASES\nFALLADOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
        with pltmain12:
            st.dataframe(df[['ActionID', 'Event', 'Minute', 'EfectiveMinute', 'PlayerID', 'Player', 'Team', 'X1', 'Y1', 'X2', 'Y2']])
    if PlotVizSelExpData == "Recuperaciones":
        pltmnop21, pltmnop22, pltmnop23 = st.columns(3)
        with pltmnop21:
            OptionPlot = ['Recoveries Map', 'Recoveries - Heatmap Bins']
            OptionPlotSel = st.selectbox('Seleccionar tipo gráfico:', OptionPlot)
        with pltmnop22:
            EfectMinSel = st.slider('Seleccionar rango de partido:', 0, MaxAddMin, (0, MaxAddMin))
        with pltmnop23:
                MetOption = ['WinStats', 'FD']
                MetOptionSel = st.selectbox('Choose color type:', MetOption)
        if MetOptionSel == 'WinStats':
            hex_list2 = ['#121214', '#D81149', '#FF0050']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#FF0050"
        if MetOptionSel == 'FD':
            hex_list2 = ['#5A9212', '#70BD0C', '#83E604']
            hex_list = ['#121214', '#545454', '#9F9F9F']
            colorviz = "#83E604"
        pltmain21, pltmain22 = st.columns(2)
        with pltmain21:
            fig, ax = mplt.subplots(figsize=(8, 8), dpi = 800)
            ax.axis("off")
            fig.patch.set_visible(False)
            pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1.0, goal_type='box', pitch_length=105, pitch_width=68)
            pitch.draw(ax=ax)
            #Adding directon arrow
            ax29 = fig.add_axes([0.368,0.22,0.3,0.05])
            ax29.axis("off")
            ax29.set_xlim(0,10)
            ax29.set_ylim(0,10)
            ax29.annotate('', xy=(2, 6), xytext=(8, 6), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            #ax29.annotate(s='', xy=(2, 5), xytext=(8, 5), arrowprops=dict(arrowstyle='<-', ls= '-', lw = 1, color = (1,1,1,0.5)))
            ax29.text(5, 2, 'Dirección campo de juego', fontproperties=prop3, c=(1,1,1,0.5), fontsize=10, ha='center')
            #Adding winstats logo
            ax53 = fig.add_axes([0.82, 0.14, 0.05, 0.05])
            url53 = "https://i.postimg.cc/R0QjGByL/sZggzUM.png"
            response = requests.get(url53)
            img = Image.open(BytesIO(response.content))
            ax53.imshow(img)
            ax53.axis("off")
            ax53.set_facecolor("#000")
            #st.dataframe(dfDOWN)
            df = df[(df['EfectiveMinute'] >= EfectMinSel[0]) & (df['EfectiveMinute'] <= EfectMinSel[1])]
            dfKK = df
            if OptionPlotSel == 'Recoveries Map':
                df = dfKK
                df = df[(df['Event'] == 'Recoveries')].reset_index()
                dfKKK = df
                dfKKK = df.drop_duplicates(subset=['X1', 'Y1', 'X2', 'Y2'], keep='last')
               
                pitch = Pitch(pitch_color='None', pitch_type='custom', line_zorder=1, linewidth=1, goal_type='box', pitch_length=105, pitch_width=68)
                pitch.draw(ax=ax)
                
                ax.scatter(df['X1'], df['Y1'], color=colorviz, edgecolors='#121214', zorder=3, lw=0.5, s=70)           

                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKK)) + " PASES COMPLETOS", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.20,0.14,0.63,0.07])
                ax9.set_xlim(0,105)
                ax9.set_ylim(0,20)
                ax9.axis("off")
                #ax9.scatter(26.25, 12, marker='s', color='#9F9F9F', s=300)
                #ax9.text(26.25, 2, 'PASES EFECTIVOS', color='#9F9F9F', fontproperties=prop2, ha='center', fontsize=9)
                #ax9.scatter(52.5, 12, marker='s', color=colorviz, s=300)
                #ax9.text(52.5, 2, 'PASES PROGRESIVOS', color=colorviz, fontproperties=prop2, ha='center', fontsize=9)
                #ax9.scatter(78.75, 12, marker='s', color='#C7B200', s=300)
                #ax9.text(78.75, 2, 'PASES CLAVES', color='#C7B200', fontproperties=prop2, ha='center', fontsize=9)

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
            elif OptionPlotSel == 'Recoveries - Heatmap Bins':
                df = dfKK
                path_eff = [path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]
                bin_statistic = pitch.bin_statistic_positional(df.X1, df.Y1, statistic='count', positional='full', normalize=True)
                pitch.heatmap_positional(bin_statistic, ax=ax, cmap=cmaps, edgecolors='#524F50', linewidth=1)
                pitch.scatter(df.X1, df.Y1, c='w', s=15, alpha=0.02, ax=ax)
                labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=14, fontproperties=prop2, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)
                #ax.text(52.5,70, "" + PlayerSelExpData.upper() + " - " + str(len(dfKKcleaned)) + " TOQUES", c='w', fontsize=10, fontproperties=prop2, ha='center')
                ax9 = fig.add_axes([0.14,0.15,0.20,0.07])
                ax9.scatter(6.75,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=1.0)
                ax9.scatter(5.00,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.6)
                ax9.scatter(3.25,5, c=colorviz, marker='h', s=400, edgecolors='#121214', alpha=0.2)
                ax9.text(5, 0, '-  ACCIONES REALIZADAS  +', c='w', fontproperties=prop2, fontsize=9, ha='center')
                ax9.axis("off")
                ax9.set_xlim(0,10)
                ax9.set_ylim(0,10)

                st.pyplot(fig, bbox_inches="tight", pad_inches=0.05, dpi=400, format="png")
                
        with pltmain22:
            st.dataframe(df[['ActionID', 'Event', 'Minute', 'EfectiveMinute', 'PlayerID', 'Player', 'Team', 'X1', 'Y1', 'X2', 'Y2']])
        
if menu_id == "DataScraping":
    with st.sidebar:
        with open("Resources/win.png", "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
            st.sidebar.markdown(
                f"""
                <div style="display:table;margin-top:-20%">
                    <img src="data:image/png;base64,{data}" width="300">
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("""---""")
        with st.form(key='form5'):
            MatchURL = st.text_input("Match URL", key="matchurl")

            DataMode = st.checkbox("Activate calculated columns")
            submit_button2 = st.form_submit_button(label='Aceptar')
    st.title("DATA SCRAPING")
    sofascore = sfc.Sofascore()
    data = sofascore.get_general_match_stats(MatchURL)
    st.dataframe(data)
