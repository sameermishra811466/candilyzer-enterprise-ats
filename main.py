import streamlit as st
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from pypdf import PdfReader
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import bcrypt
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.pdfgen import canvas

st.set_page_config(layout="wide")

# ================= THEME =================
theme = st.sidebar.toggle("🌗 Dark Mode", value=True)

if theme:
    st.markdown("""
    <style>
    .stApp {background:#0f2027;color:white;}
    label, p, span, div {color:white !important;}
    input, textarea {color:white !important;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp{background:white;color:black}</style>", unsafe_allow_html=True)

# ================= DB =================
def init_db():
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS candidates(
        name TEXT,
        role TEXT,
        score INTEGER,
        report TEXT,
        stage TEXT,
        skills TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        password TEXT,
        role TEXT
    )
    """)

    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        hashed = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        c.execute("INSERT INTO users VALUES ('admin', ?, 'admin')", (hashed,))

    conn.commit()
    conn.close()

init_db()

# ================= AUTH =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.subheader("🔐 Login")
    u = st.text_input("Username", key="login_u")
    p = st.text_input("Password", type="password", key="login_p")

    if st.button("Login"):
        conn = sqlite3.connect("candidates.db")
        c = conn.cursor()
        c.execute("SELECT password, role FROM users WHERE username=?", (u,))
        r = c.fetchone()
        conn.close()

        if r and bcrypt.checkpw(p.encode(), r[0].encode()):
            st.session_state.logged_in=True
            st.session_state.user=u
            st.session_state.role=r[1]
            st.rerun()
        else:
            st.error("Invalid login")

def signup():
    st.subheader("Signup")
    u = st.text_input("Username", key="signup_u")
    p = st.text_input("Password", type="password", key="signup_p")

    if st.button("Create"):
        conn = sqlite3.connect("candidates.db")
        c = conn.cursor()
        try:
            hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
            c.execute("INSERT INTO users VALUES (?,?,?)",(u,hashed,"recruiter"))
            conn.commit()
            st.success("Created")
        except:
            st.error("Exists")
        conn.close()

if not st.session_state.logged_in:
    t1,t2=st.tabs(["Login","Signup"])
    with t1: login()
    with t2: signup()
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.logged_in=False
    st.rerun()

st.title("🧠 Candilyzer Enterprise ATS")

# ================= GROQ =================
groq = st.sidebar.text_input("Groq API Key", type="password")

def create_model():
    if not groq:
        st.warning("Enter Groq key")
        st.stop()
    return OpenAILike(id="llama-3.1-8b-instant",api_key=groq,base_url="https://api.groq.com/openai/v1")

# ================= UTIL =================
def extract_pdf(file):
    reader = PdfReader(file)
    txt=""
    for p in reader.pages:
        if p.extract_text():
            txt+=p.extract_text()
    return txt

def extract_skills(txt):
    skills = ["Python","SQL","Machine Learning","Tableau","Power BI","Excel","Communication"]
    return [s for s in skills if s.lower() in txt.lower()]

def export_pdf(text):
    path="report.pdf"
    c=canvas.Canvas(path)
    y=800
    for line in text.split("\n"):
        c.drawString(30,y,line[:90])
        y-=15
    c.save()
    return path

# ================= NAV =================
page = st.sidebar.radio("Navigation",[
"🧠 Analyze","📊 Dashboard","📋 Kanban","🔍 Search",
"⚖️ Compare","📈 Analytics","🏆 Leaderboard","👤 Admin"
])

# ================= ANALYZE =================
if page=="🧠 Analyze":
    f=st.file_uploader("Resume",type="pdf")
    name=st.text_input("Name")
    role=st.text_input("Role")
    stage=st.selectbox("Stage",["Screening","Interview","Shortlisted","Rejected"])

    if st.button("Analyze") and f:
        txt=extract_pdf(f)
        skills=extract_skills(txt)

        agent=Agent(model=create_model(),markdown=True)
        with st.spinner("Analyzing"):
            res=agent.run(f"Analyze resume:\n{txt}\nGive score like 75/100")

        st.markdown(res.content)
        score=int(re.search(r"(\d{1,3})/100",res.content).group(1)) if re.search(r"(\d{1,3})/100",res.content) else 60

        conn=sqlite3.connect("candidates.db")
        conn.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?)",(name,role,score,res.content,stage,json.dumps(skills)))
        conn.commit()
        conn.close()

        pdf=export_pdf(res.content)
        st.download_button("Download PDF",open(pdf,"rb"),file_name="report.pdf")

# ================= DASHBOARD =================
if page=="📊 Dashboard":
    df=pd.read_sql("SELECT * FROM candidates",sqlite3.connect("candidates.db"))
    if not df.empty:
        st.metric("Total",len(df))
        st.plotly_chart(px.bar(df,x="name",y="score",color="stage"))

# ================= KANBAN =================
if page=="📋 Kanban":
    df=pd.read_sql("SELECT * FROM candidates",sqlite3.connect("candidates.db"))
    cols=st.columns(4)
    stages=["Screening","Interview","Shortlisted","Rejected"]
    for i,s in enumerate(stages):
        with cols[i]:
            st.subheader(s)
            for n in df[df["stage"]==s]["name"]:
                st.success(n)

# ================= SEARCH =================
if page=="🔍 Search":
    df=pd.read_sql("SELECT * FROM candidates",sqlite3.connect("candidates.db"))
    q=st.text_input("Search resume")
    if q:
        vect=TfidfVectorizer().fit_transform(df["report"])
        query=TfidfVectorizer().fit(df["report"]).transform([q])
        scores=(vect*query.T).toarray()
        st.write(df.iloc[scores.argmax()]["name"])

# ================= COMPARE =================
if page=="⚖️ Compare":
    df=pd.read_sql("SELECT * FROM candidates",sqlite3.connect("candidates.db"))
    c1=st.selectbox("C1",df["name"])
    c2=st.selectbox("C2",df["name"],index=1)

    if st.button("Compare"):
        d1=df[df["name"]==c1].iloc[0]
        d2=df[df["name"]==c2].iloc[0]

        st.table(pd.DataFrame({"Candidate":[c1,c2],"Score":[d1.score,d2.score]}))

        # radar
        s1=json.loads(d1.skills) if d1.skills else []
        s2=json.loads(d2.skills) if d2.skills else []

        fig=go.Figure()
        fig.add_trace(go.Scatterpolar(r=[1]*len(s1),theta=s1,fill='toself',name=c1))
        fig.add_trace(go.Scatterpolar(r=[1]*len(s2),theta=s2,fill='toself',name=c2))
        st.plotly_chart(fig)

# ================= ANALYTICS =================
if page=="📈 Analytics":
    df=pd.read_sql("SELECT * FROM candidates",sqlite3.connect("candidates.db"))
    if not df.empty:
        st.plotly_chart(px.histogram(df,x="score"))
        st.plotly_chart(px.pie(df,names="stage"))

# ================= LEADERBOARD =================
if page=="🏆 Leaderboard":
    df=pd.read_sql("SELECT * FROM candidates",sqlite3.connect("candidates.db"))
    st.dataframe(df.sort_values("score",ascending=False))

# ================= ADMIN =================
if page=="👤 Admin":
    if st.session_state.role!="admin":
        st.error("Admin only")
    else:
        users=pd.read_sql("SELECT username,role FROM users",sqlite3.connect("candidates.db"))
        st.dataframe(users)
