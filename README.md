# Social Media Bullying Detection Dashboard
An interactive data dashboard that analyzes and visualizes cyberbullying trends on Reddit. The platform also serves as an educational tool by allowing users to check if their own posts could be considered.

### **Theme**

Digital Safety & Social Impact
Using data to combat harmful online behavior and promote healthier digital spaces.

### **Tech Stack**
Python, Pandas, Scikit-learn, Transformers
RoBERTa (Fine-tuned for cyberbullying classification)
SQLite for lightweight data storage
Great Expectations for data validation
Streamlit for dashboard creation
Prefect for data pipeline orchestration

### **How to run the project**

to run the project, do the following steps:

git clone https://github.com/amnarazakhan/DE_project


#### _setting up a virtual env:_

python -m venv venv

source venv/bin/activate   # On Windows: venv\Scripts\activate

#### _Installing dependencies:_

pip install -r requirements.txt


After making sure the requirements are installed, run:

streamlit run app.py

