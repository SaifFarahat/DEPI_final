import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# deleting libraries that won't be used
import warnings 
warnings.filterwarnings("ignore")

# ML libraries
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier





# Header

# Header و HTML + CSS
st.markdown("""
    <style>
    .nav-container {
        background-color: #333;
        padding: 10px;
        display: flex;
        justify-content: center;
    }
    .nav-button {
        background-color: #ff4d4d;
        border: none;
        color: white;
        padding: 10px 20px;
        margin: 0 10px;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .nav-button:hover {
        background-color: #ff4d4d;
    }
    </style>
    <div class="nav-container">
        <a href="/?page=Home" target="_self"><button class="nav-button">Home</button></a>
        <a href="/?page=Visualization" target="_self"><button class="nav-button">Visualization</button></a>
        <a href="/?page=Model" target="_self"><button class="nav-button">Model</button></a>
        <a href="/?page=About" target="_self"><button class="nav-button">ِAbout</button></a>
    </div>
""", unsafe_allow_html=True)

# استخدام st.query_params بدل experimental
query_params = st.query_params
page = query_params.get("page", "Home")  # load file










# fixed for all pages

# Inject CSS to make the sidebar 10% transparent
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.1);  /* white with 10% opacity */
        }
    </style>
    """, unsafe_allow_html=True)




# Inject CSS to make the text float at the bottom-right and stay fixed while scrolling
st.markdown("""
    <style>
        .float-text {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 14px;  /* Smaller font size */
            color: white;  /* Golden color */
            z-index: 9999;  /* Make sure it's on top of other elements */
            text-align: right;  /* Align the text to the right */
        }
    </style>
    <div class="float-text">
        This app developed by:
        <br>
        Eng Saif
        <br>
        Eng Emad
        <br>
        Eng Elsayed
    </div>
    """, unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://static.vecteezy.com/system/resources/thumbnails/013/129/253/small_2x/stethoscope-on-a-glass-table-dark-background-the-concept-of-medicine-and-medical-treatment-photo.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<h1 style='text-align: center; color: #ff4d4d;'>Healthcare Predictive Analysis</h1><br><hr>",
    unsafe_allow_html=True
)



# load file

#cache
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


file=st.sidebar.file_uploader("Upload file", type=["csv"])
















if page == "Home":

    # Set the background image for the About page
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://eu-images.contentstack.com/v3/assets/blt6b0f74e5591baa03/blt6b328cb4d6ee36cd/670d6546881bc37741dcb165/GettyImages-1946361629.jpg?width=1280&auto=webp&quality=95&format=jpg&disable=upscale');
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("Welcome to the Healthcare Prediction App")
    
    st.write("""
    This app is designed for interactive data visualization and creating dashboards. 
    It allows users to easily experiment with different models on a dataset, facilitating 
    the comparison of various model performance metrics such as accuracy, recall, F1 score, and more.
    """)
    
    st.write("""
    Additionally, you can modify or delete parameters or features and observe how the model responds to these changes.
    """)
    
    st.write("""
    The main purpose of this application is to provide healthcare predictions based on the information you input.
    It predicts your health condition and potential diseases that may affect you in the future.
    """)










# Visualization page
elif page == "Visualization":

    if file is not None:
        df = load_data(file)
        n_rows = st.slider('Choose number of rows to display', min_value=5, max_value=len(df), step=1)

        columns_to_show = st.multiselect("Select columns to show", df.columns.to_list(), default=df.columns.to_list())

        st.write(df[:n_rows][columns_to_show])

        # Corrected: Add one more variable for the "Pair Plot" tab
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["Box", "Histogram", "HeatMap", "Scatter", "Line Plot", "Bar Plot","Pair Plot"]
        )

        numerical_columns = df.select_dtypes(include=np.number).columns.to_list()

        # Box Plot
        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                x_axis = st.selectbox("Select X axis", numerical_columns, key="x_axis_box")

            with col2:
                y_axis = st.selectbox('Select y axis', numerical_columns, key="y_axis_box")

            if x_axis is not None:
                fig_box = px.box(df, y=x_axis)
                with st.spinner("Computing..."):  # spinner
                    st.plotly_chart(fig_box)

        # Histogram
        with tab2:
            histogram_feature = st.selectbox('Select feature for histogram', numerical_columns, key="histogram_feature")
            if histogram_feature is not None: 
                fig_hist = px.histogram(df, x=histogram_feature)
                st.plotly_chart(fig_hist)

        # HeatMap
        with tab3:
            corr_matrix = df.corr()
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto='.1f',
                aspect="auto",
                color_continuous_scale='Viridis'
            )
            fig_heatmap.update_layout(width=800, height=600)
            st.plotly_chart(fig_heatmap, key="heatmap")

        # Scatter Plot
        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                x_axis = st.selectbox("Select X axis", numerical_columns, key="x_axis_scatter")

            with col2:
                y_axis = st.selectbox('Select Y axis', numerical_columns, key="y_axis_scatter")

            if x_axis is not None and y_axis is not None:
                fig_scatter = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter plot of {x_axis} vs {y_axis}')
                with st.spinner("Computing..."):
                    st.plotly_chart(fig_scatter)

        # Line Plot
        with tab5:
            line_feature = st.selectbox("Select feature for line plot", numerical_columns, key="line_feature")
            if line_feature is not None:
                fig_line = px.line(df, y=line_feature, title=f'Line plot of {line_feature}')
                with st.spinner("Computing..."):
                    st.plotly_chart(fig_line)

        # Bar Plot
        with tab6:
            bar_feature = st.selectbox("Select feature for bar plot", numerical_columns, key="bar_feature")
            if bar_feature is not None:
                fig_bar = px.bar(df, x=df.index, y=bar_feature, title=f'Bar plot of {bar_feature}')
                with st.spinner("Computing..."):
                    st.plotly_chart(fig_bar)


        # Pair Plot (Using Seaborn)
        with tab7:
            sns.set(style="whitegrid")
            pair_feature = st.multiselect("Select features for pair plot", df.select_dtypes(include=np.number).columns.tolist(), key="pair_feature")
            if len(pair_feature) > 1:
                fig_pair = sns.pairplot(df[pair_feature])
                st.pyplot(fig_pair)













elif page == "Model":


    # دالة لتحميل البيانات
    def load_data(file):
        return pd.read_csv(file)

    # فحص توازن البيانات
    def check_imbalance(y, threshold=0.2):
        counts = y.value_counts()
        proportions = y.value_counts(normalize=True)
        min_ratio = proportions.min()

        st.write("\nAbsolute Class Counts:")
        st.write(counts)
        
        st.write("\nClass Proportions:")
        st.write(proportions)
        
        if min_ratio < threshold:
            st.warning("\n⚠️ The label is imbalanced!")
            return 'imbalanced'
        else:
            st.success("\n✅ The label is balanced.")
            return 'balanced'
        
        # رسم توزيع الفئات
        plt.figure(figsize=(6,4))
        sns.countplot(x=y)
        plt.title("Class Distribution")
        st.pyplot(plt)  # استخدم st.pyplot لعرض الرسم في Streamlit

    # واجهة المستخدم باستخدام Streamlit
    st.title("Data Analysis App")

    # file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        df = load_data(file)

        # تأكد من أن 'features' و 'target' يتم تحديدهما بشكل صحيح
        features = st.multiselect("Select feature columns", df.columns.to_list(), default=df.columns.to_list())
        set1 = set(df.columns.to_list())
        set2 = set(features)

        unique_elements = list(set1.symmetric_difference(features))

        # تأكد من اختيار عمود واحد فقط كـ target
        target = st.multiselect("Select Target column", unique_elements)

        # التأكد من أن المستخدم قد اختار عمودًا واحدًا كـ target
        if len(target) == 1:
            y = df[target[0]]  # اختر العمود الوحيد في target
        else:
            st.error("Please select exactly one target column.")
        


        # فحص توازن البيانات
        if len(target) == 1:
            check = check_imbalance(y)
        else:
            st.warning('Please select exactly one target column to check imbalance.')
            

        




                # التأكد من أن الأعمدة المختارة موجودة
        if features and target:
            X = df[features]  # استخدام الأعمدة المحددة للـ features
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



        
        # handel imbalanced data
        if check=='imbalanced':
            st.write('SMOTE will be used to handle imbalanced data.')


            X = df[features]
            smote_method = SMOTE()
            fearures_smote , label_smote = smote_method.fit_resample(X,y)
            label_smote.value_counts()
            x_train, x_test, y_train, y_test = train_test_split(fearures_smote, label_smote, test_size= 0.25)







        tab1, tab2, tab3, tab4, tab5, tab6, tab7=st.tabs(["DecisionTree", "RandomForest", "LogisticRegression","KNN","XGboost","SVM","catboost"])


        #INPUT
        def user_input(features):
            data = {}
            st.sidebar.write('Please enter your data for prediction.')
            for col in features.columns:
                value = st.sidebar.slider(col, int(features[col].min()), int(features[col].max()), step=1)
                data[col] = value

            input_df = pd.DataFrame([data])
            return input_df
        input_df = user_input(X)







            
        with tab1:

            if "model" not in st.session_state:
                st.session_state.model = None

            if st.button("Apply Model", key="apply_model_tab1"):
                if len(target) == 1:
                    # كان وحش
                    # model = DecisionTreeClassifier(max_depth=5, random_state=42)  # يمكنك تغيير max_depth أو إضافة معلمات أخرى
                    # تقريبا حفط الداتا
                    model = DecisionTreeClassifier(max_depth=11, random_state=42)  # يمكنك تغيير max_depth أو إضافة معلمات أخرى
                    model.fit(x_train, y_train)

                    st.session_state.model = model

                    y_pred = model.predict(x_test)
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()

                    st.subheader('Classification Report')
                    st.dataframe(report_df.style.format("{:.2f}"))
                else:
                    st.error('Please choose one column to be the target')


            if st.button("Predict", key="predict_tab1"):
                if (
                    st.session_state.model is not None and 
                    hasattr(st.session_state.model, "predict")
                ):
                    prediction = st.session_state.model.predict(input_df)
                    st.write('Prediction: ', prediction) 
                    print(prediction)
                else:
                    st.error("Please apply the model first by clicking 'Apply Model'.")










        with tab2:


            if "model" not in st.session_state:
                st.session_state.model = None

            if st.button("Apply Model", key="apply_model_tab2"):
                if len(target) == 1:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(x_train, y_train)

                    st.session_state.model = model

                    y_pred = model.predict(x_test)
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()

                    st.subheader('Classification Report')
                    st.dataframe(report_df.style.format("{:.2f}"))
                else:
                    st.error('Please choose one column to be the target')


            if st.button("Predict", key="predict_tab2"):
                if (
                    st.session_state.model is not None and 
                    hasattr(st.session_state.model, "n_estimators") and 
                    hasattr(st.session_state.model, "estimators_")
                ):
                    prediction = st.session_state.model.predict(input_df)
                    st.write('Prediction: ', prediction) 
                    print(prediction)
                else:
                    st.error("Please apply the model first by clicking 'Apply Model'.")




        with tab3:

            if "model" not in st.session_state:
                st.session_state.model = None

            if st.button("Apply Model", key="apply_model_tab3"):
                if len(target) == 1:
                    model = LogisticRegression(max_iter=1000, random_state=42)
                    model.fit(x_train, y_train)

                    st.session_state.model = model

                    y_pred = model.predict(x_test)
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()

                    st.subheader('Classification Report')
                    st.dataframe(report_df.style.format("{:.2f}"))
                else:
                    st.error('Please choose one column to be the target')


            if st.button("Predict", key="predict_tab3"):
                if (
                    st.session_state.model is not None and 
                    hasattr(st.session_state.model, "coef_")
                ):
                    prediction = st.session_state.model.predict(input_df)
                    st.write('Prediction: ', prediction) 
                    print(prediction)
                else:
                    st.error("Please apply the model first by clicking 'Apply Model'.")



        with tab4:
            if "model" not in st.session_state:
                st.session_state.model = None

            if st.button("Apply Model", key="apply_model_tab4"):
                if len(target) == 1:
                    model = KNeighborsClassifier(n_neighbors=5)  # اختر عدد الجيران (n_neighbors) كما تريد
                    model.fit(x_train, y_train)

                    st.session_state.model = model

                    y_pred = model.predict(x_test)
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()

                    st.subheader('Classification Report')
                    st.dataframe(report_df.style.format("{:.2f}"))
                else:
                    st.error('Please choose one column to be the target')

            if st.button("Predict", key="predict_tab4"):
                if (
                    st.session_state.model is not None and 
                    hasattr(st.session_state.model, "predict")
                ):
                    prediction = st.session_state.model.predict(input_df)
                    st.write('Prediction: ', prediction) 
                    print(prediction)
                else:
                    st.error("Please apply the model first by clicking 'Apply Model'.")




        with tab5:

            if "model" not in st.session_state:
                st.session_state.model = None

            if st.button("Apply Model", key="apply_model_tab5"):
                if len(target) == 1:
                    model = XGBClassifier(random_state=42)  # نموذج XGBoost مع إعداد random_state للتكرار
                    model.fit(x_train, y_train)

                    st.session_state.model = model

                    y_pred = model.predict(x_test)
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()

                    st.subheader('Classification Report')
                    st.dataframe(report_df.style.format("{:.2f}"))
                else:
                    st.error('Please choose one column to be the target')


            if st.button("Predict", key="predict_tab5"):
                if (
                    st.session_state.model is not None and 
                    hasattr(st.session_state.model, "predict")
                ):
                    prediction = st.session_state.model.predict(input_df)
                    st.write('Prediction: ', prediction) 
                    print(prediction)
                else:
                    st.error("Please apply the model first by clicking 'Apply Model'.")



        
        with tab6:

            if "model" not in st.session_state:
                st.session_state.model = None

            if st.button("Apply Model", key="apply_model_tab6"):
                if len(target) == 1:
                    model = SVC(kernel='linear', random_state=42)  # اختر kernel حسب حاجتك (مثل 'linear', 'rbf', 'poly')
                    model.fit(x_train, y_train)

                    st.session_state.model = model

                    y_pred = model.predict(x_test)
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()

                    st.subheader('Classification Report')
                    st.dataframe(report_df.style.format("{:.2f}"))
                else:
                    st.error('Please choose one column to be the target')

            if st.button("Predict", key="predict_tab6"):
                if (
                    st.session_state.model is not None and 
                    hasattr(st.session_state.model, "predict")
                ):
                    prediction = st.session_state.model.predict(input_df)
                    st.write('Prediction: ', prediction) 
                    print(prediction)
                else:
                    st.error("Please apply the model first by clicking 'Apply Model'.")


        
        with tab7:

            if "model" not in st.session_state:
                st.session_state.model = None

            if st.button("Apply Model", key="apply_model_tab7"):
                if len(target) == 1:
                    model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, random_state=42, verbose=0)  # ضبط المعلمات كما تريد
                    model.fit(x_train, y_train)

                    st.session_state.model = model

                    y_pred = model.predict(x_test)
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()

                    st.subheader('Classification Report')
                    st.dataframe(report_df.style.format("{:.2f}"))
                else:
                    st.error('Please choose one column to be the target')

            if st.button("Predict", key="predict_tab7"):
                if (
                    st.session_state.model is not None and 
                    hasattr(st.session_state.model, "predict")
                ):
                    prediction = st.session_state.model.predict(input_df)
                    st.write('Prediction: ', prediction) 
                    print(prediction)
                else:
                    st.error("Please apply the model first by clicking 'Apply Model'.")







elif page == "About":

    # Set the background image for the About page
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://img.freepik.com/premium-photo/abstract-technology-wave-particles-big-data-visualization-dark-background-with-motion-dots-lines-artificial-intelligence-3d-rendering_744733-697.jpg');
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.title("About This Page")
    
    st.write("""
    This page has been created by students from the "Digital Egypt Pioneers initiative", 
    Track: IBM AI & Data Science.
    
    **Team Members:**
    - Saif Farahat (Team Leader)
    - Emadeldin Waleed
    - Elsayed
    
    If you'd like to get in touch, here is our contact information:
    """)
    
    st.write("""
    **Name:** Saif Mahmod Farahat (Full Name)  
    **Phone Number:** 01140764360  
    **Email:** saifmahmoude@gmail.com  
    **LinkedIn:** [Saif Mahmod Farahat](https://www.linkedin.com/in/saif-mahmoud-830b2527b/)
    """)

    st.write("""
        **Name:** Emadeldin Waleed Abdulsalam (Full Name)  
        **Phone Number:** 01559017222 
        **Email:** emadwaleed1104@gmail.com 
        **LinkedIn:** [Emad Waleed](https://www.linkedin.com/in/emad-waleed/)
        """)
    st.write("""
        **Name:** Elsayed Elsherbiny(Full Name)  
        **Phone Number:** 01097742883
        **LinkedIn:** [Elsayed Elsherbiny](https://www.linkedin.com/in/elsayed11/)
        """)
    st.write("""
    Feel free to reach out for any collaborations or inquiries.
    """)