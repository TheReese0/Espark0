import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import mesa
import json
import math
import time
import datetime
import numpy as np
from openai import OpenAI

# ==============================================================================
# 1. 页面配置与 CSS (严格保持侧边栏 320px 设计)
# ==============================================================================
st.set_page_config(
    page_title="Espark Policy Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* --- 全局配色 --- */
    .stApp { background-color: #0b0f19; color: #e0e6ed; }
    
    /* --- 侧边栏深度定制 (严格保持 320px) --- */
    [data-testid="stSidebar"] {
        background-color: #10141d;
        border-right: 1px solid #2d333b;
        min-width: 320px !important;
        max-width: 320px !important;
        padding-top: 0px;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0px;
    }
    
    /* Espark Header */
    .sidebar-header {
        padding: 30px 20px;
        background: linear-gradient(135deg, rgba(77, 107, 254, 0.15) 0%, rgba(16, 20, 29, 0) 100%);
        border-bottom: 1px solid #2d333b;
        margin-bottom: 0px;
    }
    .sidebar-logo { font-size: 32px; font-weight: 900; color: white; font-family: 'Arial Black', sans-serif; letter-spacing: -1px; }
    .sidebar-sub { font-size: 10px; color: #4d6bfe; font-weight: bold; letter-spacing: 2px; text-transform: uppercase; margin-top: 5px; }
    
    /* --- 导航菜单样式 --- */
    .stRadio > label { display: none; }
    div[role="radiogroup"] { padding: 20px 10px; }
    div[role="radiogroup"] label > div:first-child { display: none; }
    
    div[role="radiogroup"] label {
        padding: 12px 15px !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
        border: 1px solid transparent;
        transition: all 0.2s;
        background: transparent;
        color: #8b949e;
        font-weight: 500;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.05);
        color: white;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: rgba(77, 107, 254, 0.15) !important;
        border: 1px solid rgba(77, 107, 254, 0.3) !important;
        color: #4d6bfe !important;
        font-weight: bold;
    }
    
    /* --- 按钮样式 --- */
    div.stButton > button {
        background: #4d6bfe; color: white; border: none; height: 45px; font-size: 15px; font-weight: 600; border-radius: 6px; box-shadow: 0 4px 12px rgba(77, 107, 254, 0.3);
    }
    div.stButton > button:hover { background: #3b5bdb; transform: translateY(-1px); }
    
    /* --- 历史记录折叠卡片 (交互核心) --- */
    .streamlit-expanderHeader {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #e6edf3;
        font-weight: 600;
        font-size: 15px;
    }
    .streamlit-expanderContent {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 20px;
    }
    
    /* --- 实时日志卡片 (高亮) --- */
    .latest-card {
        background: #1c2128;
        border-left: 4px solid #4d6bfe;
        padding: 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心逻辑 (保持 v7 Strategic 内核)
# ==============================================================================
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = [] 

class StrategicAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.policy_stage = 0 
        self.policy_names = ["严格一孩", "试点(双独/单独)", "全面二孩", "三孩及配套"]
        
    def step(self):
        year = self.model.year
        current_pol = self.policy_names[self.policy_stage]
        economy_context = self.model.get_economic_context(year)
        labor_status = self.model.get_labor_supply_status(year)
        grassroots = self.model.get_grassroots_feedback(year)
        
        thought = "模拟推演中..."
        new_stage = self.policy_stage

        if self.model.api_key:
            try:
                if year % 2 == 0 or year > 2010:
                    user_prompt = f"""
                    【年份】{year} 【国策】{current_pol}
                    【情报】经济:{economy_context} | 劳动力:{labor_status} | 基层:{grassroots}
                    【任务】决定明年政策(0-3)。
                    【输出JSON】{{"thought": "...", "decision_code": int}}
                    """
                    client = OpenAI(api_key=self.model.api_key, base_url="https://api.deepseek.com")
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "system", "content": self.model.system_prompt}, {"role": "user", "content": user_prompt}],
                        temperature=self.model.temperature, max_tokens=300
                    )
                    content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
                    result = json.loads(content)
                    new_stage = int(result["decision_code"])
                    thought = result["thought"]
            except Exception as e:
                thought = f"AI Error: {e}"
        else:
            if year >= 2013 and self.policy_stage == 0: new_stage = 1; thought = "[模拟] 劳动力拐点显现，启动试点。"
            elif year >= 2016 and self.policy_stage == 1: new_stage = 2; thought = "[模拟] 全面二孩时刻。"
            elif year >= 2021 and self.policy_stage == 2: new_stage = 3; thought = "[模拟] 三孩时代。"

        if new_stage > self.policy_stage: self.policy_stage = new_stage
        
        return {"Year": year, "Policy": self.policy_names[self.policy_stage], "Policy_Code": self.policy_stage, 
                "Economy": economy_context, "Labor_Lag": labor_status, "Thought": thought}

class StrategicModel(mesa.Model):
    def __init__(self, api_key, system_prompt, temperature, start_year):
        super().__init__()
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.year = start_year
        self.agent = StrategicAgent("Gov", self)

    def get_economic_context(self, year):
        if year < 2000: return "经济起飞期"
        elif year < 2010: return "WTO黄金期"
        elif year < 2015: return "新常态转折点"
        else: return "高质量发展期"

    def get_labor_supply_status(self, year):
        birth_year = year - 20
        if birth_year < 1975: return "充沛"
        elif birth_year < 1990: return "充足"
        else: return "严重短缺"

    def get_grassroots_feedback(self, year):
        return "执行难度大" if year < 2000 else "群众意愿低迷"

    def step(self):
        res = self.agent.step()
        self.year += 1
        return res

def render_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Year'], y=df['Policy_Code'], 
        mode='lines', name='Policy Level', 
        fill='tozeroy', fillcolor='rgba(77, 107, 254, 0.15)',
        line=dict(color='#4d6bfe', width=3, shape='hv')
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(showgrid=False), 
        yaxis=dict(showgrid=True, gridcolor='#333', tickvals=[0,1,2,3], ticktext=["一孩","试点","二孩","三孩"])
    )
    return fig

# ==============================================================================
# 3. 侧边栏布局 (保持 320px 铺满设计 - 严格不动)
# ==============================================================================
with st.sidebar:
    # 顶部品牌区
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">Espark</div>
        <div class="sidebar-sub">INTELLIGENCE LAB v10.0</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 装饰图表 (铺满)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.random.rand(100)
    fig_net = go.Figure(go.Scatter(x=x, y=y, line=dict(color='#4d6bfe', width=1), fill='tozeroy', fillcolor='rgba(77, 107, 254, 0.1)'))
    fig_net.update_layout(height=80, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig_net, use_container_width=True, config={'displayModeBar': False})
    
    # 导航菜单
    menu = st.radio(
        "Menu", 
        ["🛠️ 智能沙盘 (Playground)", "📜 输出记录 (Logs)", "⚙️ 核心逻辑 (Core)", "🌐 市场对标 (Market)", "📚 智能体科普 (About)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🔑 Global Config")
    api_key_input = st.text_input("DeepSeek API Key", type="password")

# ==============================================================================
# 4. 主界面内容
# ==============================================================================

# --- 场景 1：核心沙盘 (Playground) ---
if menu == "🛠️ 智能沙盘 (Playground)":
    
    st.markdown("# ⚡ Espark 战略决策沙盘")
    
    # --- A. 配置区 ---
    with st.expander("🎛️ 新建推演配置 (New Simulation)", expanded=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            gov_style = st.selectbox("决策者人设", ["稳健型 (历史真实)", "激进改革型", "僵化保守型"])
            if gov_style.startswith("稳健"):
                default_prompt = "你是一个对历史负责的战略家。深知'人口政策有20年滞后性'。坚持民主集中制，不被短期民意裹挟。"
                temp = 0.3
            elif gov_style.startswith("激进"):
                default_prompt = "你是一个极具前瞻性的改革家。高度关注'20年后的劳动力危机'，一旦发现异常，宁可牺牲当下经济也要提前改革。"
                temp = 0.7
            else:
                default_prompt = "你是一个短视的决策者。只关注当下的GDP增长，完全忽略20年后的劳动力隐患。"
                temp = 0.1
            sys_prompt = st.text_area("System Prompt", value=default_prompt, height=70)
        with c2:
            temperature = st.slider("思维活跃度", 0.0, 1.0, temp)
            sim_years = st.number_input("推演年数", 20, 50, 35)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 启动新推演")

    # --- B. 运行区 ---
    if run_btn:
        st.divider()
        st.subheader("🔥 正在推演 (Live Simulation)")
        
        live_dash, live_log = st.columns([6, 4])
        with live_dash:
            chart_placeholder = st.empty()
        with live_log:
            log_placeholder = st.empty()
        
        current_run_data = []
        model = StrategicModel(api_key_input, sys_prompt, temperature, 1990)
        progress = st.progress(0)
        
        for i in range(sim_years):
            step_data = model.step()
            current_run_data.append(step_data)
            df_live = pd.DataFrame(current_run_data)
            
            # 实时图表
            chart_placeholder.plotly_chart(render_chart(df_live), use_container_width=True)
            
            # 实时日志 (最新置顶 + 历史收纳)
            with log_placeholder.container():
                # 1. 高亮显示最新日志
                latest = step_data
                pol_color = "#4d6bfe" if latest['Policy_Code'] > 0 else "#666"
                st.markdown(f"""
                <div class="latest-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-weight:bold; color:white; font-size:1.1em;">🔥 Year {latest['Year']} 决策中枢</span>
                        <span style="background:{pol_color}; padding:2px 8px; border-radius:4px; font-size:12px;">{latest['Policy']}</span>
                    </div>
                    <div style="color:#ddd; font-family:'Courier New'; font-size:0.9em;">{latest['Thought']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 2. 历史日志折叠收纳
                if len(current_run_data) > 1:
                    with st.expander(f"📚 查看过往 {len(current_run_data)-1} 条记录", expanded=False):
                        # 倒序遍历
                        for log in reversed(current_run_data[:-1]):
                            st.markdown(f"""
                            <div style="border-bottom:1px solid #333; padding:8px 0;">
                                <span style="color:#4d6bfe; font-weight:bold;">{log['Year']}</span> 
                                <span style="color:#888;">{log['Policy']}</span><br>
                                <span style="color:#888; font-size:0.85em;">{log['Thought'][:50]}...</span>
                            </div>
                            """, unsafe_allow_html=True)

            time.sleep(0.05)
            progress.progress((i+1)/sim_years)
        
        # 归档
        run_id = len(st.session_state.simulation_history) + 1
        st.session_state.simulation_history.insert(0, {
            'id': run_id,
            'time': datetime.datetime.now().strftime("%H:%M:%S"),
            'style': gov_style,
            'df': pd.DataFrame(current_run_data)
        })
        st.success("推演完成，结果已归档。")
        time.sleep(1)
        st.rerun()

    # --- C. 历史档案区 (交互核心：点击了解) ---
    if st.session_state.simulation_history:
        st.divider()
        st.subheader("📂 历史推演档案 (Interactive Archive)")
        st.caption("点击下方卡片，展开查看过往推演的战略态势和详细思维链。")
        
        for run in st.session_state.simulation_history:
            # 交互式折叠卡片
            with st.expander(f"Run #{run['id']} | {run['style']} | 🕒 {run['time']}", expanded=(run['id']==len(st.session_state.simulation_history))):
                
                h_col1, h_col2 = st.columns([6, 4])
                
                # 左侧：战略监测态势
                with h_col1:
                    st.markdown("#### 📉 战略态势回放")
                    st.plotly_chart(render_chart(run['df']), use_container_width=True, key=f"c_{run['id']}")
                    
                    csv = run['df'].to_csv(index=False).encode('utf-8-sig')
                    st.download_button(f"📥 导出 Run #{run['id']} 数据", csv, f"sim_{run['id']}.csv", "text/csv")
                
                # 右侧：决策思维链条 (带滚动条)
                with h_col2:
                    st.markdown("#### 💬 完整决策思维链")
                    log_html = "<div style='max-height: 350px; overflow-y: auto; padding-right:5px;'>"
                    for index, row in run['df'].iterrows():
                        p_color = "#4d6bfe" if row['Policy_Code'] > 0 else "#666"
                        log_html += f"""
                        <div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px; margin-bottom:8px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <span style="color:#4d6bfe; font-weight:bold;">{row['Year']}</span>
                                <span style="background:{p_color}; color:white; padding:2px 6px; border-radius:4px; font-size:10px;">{row['Policy']}</span>
                            </div>
                            <div style="color:#ccc; font-size:0.85em;">{row['Thought']}</div>
                        </div>
                        """
                    log_html += "</div>"
                    st.markdown(log_html, unsafe_allow_html=True)

# --- 场景 2：输出记录 ---
elif menu == "📜 输出记录 (Logs)":
    st.markdown("# 📜 全局数据中心")
    if st.session_state.simulation_history:
        all_dfs = [run['df'].assign(RunID=run['id']) for run in st.session_state.simulation_history]
        full_df = pd.concat(all_dfs)
        st.dataframe(full_df, use_container_width=True)
    else:
        st.info("暂无数据")

# --- 场景 3：核心逻辑 (深度理论版) ---
elif menu == "⚙️ 核心逻辑 (Core)":
    st.markdown("# ⚙️ Espark Policy Lab 核心逻辑")
    st.markdown("### 基于间断均衡与复杂自适应理论的生成式政策模拟平台")
    
    with st.container():
        st.markdown("""
        <div style='background: rgba(22, 27, 34, 0.6); border: 1px solid #30363d; border-radius: 12px; padding: 25px; margin-bottom: 20px;'>
        <p style='color: #e6edf3; font-size: 1.05em; line-height: 1.6;'>
        <strong>Espark Policy Lab</strong> 的核心逻辑深植于两大理论基石：<strong>间断均衡理论（Punctuated Equilibrium Theory）</strong>与<strong>复杂自适应系统理论（Complex Adaptive Systems Theory）</strong>。本平台并非简单的政策效果预测工具，而是一个旨在再现政策系统动态演化过程与决策者认知机制的"生成式战略沙盘"。
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 理论基础部分
    with st.expander("📚 一、理论基础：两大理论框架的融合", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: rgba(16, 20, 29, 0.8); padding: 20px; border-radius: 8px; border-left: 4px solid #4d6bfe;'>
            <h4 style='color: #ff5252; margin-top: 0;'>1. 间断均衡理论的政策过程再现</h4>
            <p style='color: #c9d1d9; font-size: 0.95em;'>
            间断均衡理论认为，政策系统长期处于稳定状态，偶因焦点事件、外部冲击或内部压力累积而爆发剧烈变革，形成"长期均衡"与"短期突变"交替的节律。在 Espark 中，这一理论体现为：
            </p>
            <ul style='color: #c9d1d9; font-size: 0.9em;'>
            <li><strong>政策阶段锁定</strong>：模型中的政策（一孩、试点、二孩、三孩）在多数年份保持稳定，模拟制度惯性。</li>
            <li><strong>压力阈值触发</strong>：当经济、劳动力、社会反馈等多维压力值突破临界点，系统便跃迁至新的政策阶段，再现"政策间断"。</li>
            <li><strong>路径依赖</strong>：每一次间断都受历史路径约束，前期政策选择限定了后续变革的空间与方向。</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: rgba(16, 20, 29, 0.8); padding: 20px; border-radius: 8px; border-left: 4px solid #00e676;'>
            <h4 style='color: #00e676; margin-top: 0;'>2. 复杂自适应系统的涌现与适应</h4>
            <p style='color: #c9d1d9; font-size: 0.95em;'>
            复杂自适应系统理论强调，系统由多个相互作用的适应性主体构成，通过自组织、学习和反馈涌现出宏观模式。Espark 将其具象化为：
            </p>
            <ul style='color: #c9d1d9; font-size: 0.9em;'>
            <li><strong>自适应主体</strong>：生成式智能体作为核心决策者，能够根据环境变化调整认知与策略，具备"学习"与"适应"能力。</li>
            <li><strong>多层次互动</strong>：微观的个体决策（智能体）与宏观的经济、人口、社会压力持续互动，形成双向反馈。</li>
            <li><strong>非线性涌现</strong>：政策结果并非简单加总，而是系统各要素非线性相互作用下涌现的宏观态势，具有不可完全预测性。</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # 核心机制部分
    st.markdown("---")
    st.markdown("### 二、核心机制：跨代际延迟反馈的认知仿真")
    
    st.markdown("""
    <div style='background: rgba(22, 27, 34, 0.6); border: 1px solid #30363d; border-radius: 12px; padding: 25px; margin-bottom: 20px;'>
    <p style='color: #e6edf3; font-size: 1.05em; line-height: 1.6;'>
    在上述理论指导下，Espark 构建了一个"感知—评估—决策—反馈"的闭环认知仿真系统：
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 机制细节
    tabs = st.tabs(["延迟反馈感知", "多元压力评估", "生成式认知", "间断式跃迁"])
    
    with tabs[0]:
        st.markdown("""
        <div style='background: rgba(30, 35, 45, 0.6); padding: 20px; border-radius: 8px; height: 100%;'>
        <h4 style='color: #4d6bfe; margin-top: 0;'>1. 延迟反馈感知机制</h4>
        <p style='color: #c9d1d9;'>
        模型设定了一个根本性约束：<strong>今日劳动力供给由二十年前出生政策决定</strong>。这一"20年滞后"机制将决策的长期后果具象化为实时可感知的压力信号，迫使智能体必须进行跨代际的前瞻思考，直面短期政治经济压力与长期人口安全之间的根本矛盾。
        </p>
        <div style='background: #0d1117; padding: 15px; border-radius: 6px; margin-top: 15px; font-family: monospace;'>
        <span style='color: #58a6ff;'># 核心算法：跨代际反馈</span><br>
        <span style='color: #79c0ff;'>birth_year</span> = <span style='color: #ff7b72;'>year</span> - <span style='color: #a5d6ff;'>20</span><br>
        <span style='color: #ffa657;'># 今天的劳动力 = 20年前的出生人口</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("""
        <div style='background: rgba(30, 35, 45, 0.6); padding: 20px; border-radius: 8px; height: 100%;'>
        <h4 style='color: #4d6bfe; margin-top: 0;'>2. 多元压力评估框架</h4>
        <p style='color: #c9d1d9;'>
        智能体持续监测多条并行的"信息流"：
        </p>
        <ul style='color: #c9d1d9;'>
        <li><span style='color: #ff5252'>经济流</span>：宏观经济阶段定性（如"WTO黄金期"、"新常态转折点"），代表发展的即时需求。</li>
        <li><span style='color: #00e676'>人口流</span>：基于滞后机制的劳动力预警（"充沛"→"趋紧"→"严重短缺"），代表未来的结构性危机。</li>
        <li><span style='color: #ffb74d'>政治流</span>：基层执行反馈与民意倾向，代表社会的承受力与反应。</li>
        </ul>
        <p style='color: #c9d1d9;'>
        多元压力的汇聚、冲突与优先级竞争，构成了决策的张力场。
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("""
        <div style='background: rgba(30, 35, 45, 0.6); padding: 20px; border-radius: 8px; height: 100%;'>
        <h4 style='color: #4d6bfe; margin-top: 0;'>3. 生成式认知决策过程</h4>
        <p style='color: #c9d1d9;'>
        区别于传统模型的规则驱动，Espark 的智能体通过大语言模型进行情境化推理：
        </p>
        <ul style='color: #c9d1d9;'>
        <li><strong>记忆与反思</strong>：参考历史政策效果，形成路径依赖。</li>
        <li><strong>权衡与博弈</strong>：在不同压力流之间进行价值排序与风险权衡。</li>
        <li><strong>风格化输出</strong>：依据预设的"人设"（稳健、激进、保守），同一情境下可能产生不同的决策逻辑与时机选择。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("""
        <div style='background: rgba(30, 35, 45, 0.6); padding: 20px; border-radius: 8px; height: 100%;'>
        <h4 style='color: #4d6bfe; margin-top: 0;'>4. 间断式政策跃迁</h4>
        <p style='color: #c9d1d9;'>
        当压力累积突破系统阈值，智能体推动政策阶段发生跃迁（如"试点→全面二孩"）。这种跃迁并非平滑渐进，而是系统在长期僵局后为应对危机而重构的"间断均衡点"，符合政策变迁的真实历史节律。
        </p>
        <div style='background: #0d1117; padding: 15px; border-radius: 6px; margin-top: 15px; font-family: monospace;'>
        <span style='color: #58a6ff;'># 间断式跃迁算法</span><br>
        <span style='color: #79c0ff;'>if</span> <span style='color: #ff7b72;'>压力值</span> > <span style='color: #a5d6ff;'>阈值</span>:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;<span style='color: #79c0ff;'>政策阶段</span> = <span style='color: #a5d6ff;'>下一阶段</span><br>
        <span style='color: #ffa657;'># 模拟政策间断</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 模型价值部分
    st.markdown("---")
    st.markdown("### 三、模型价值：从解释过去到探索可能")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        <div style='background: rgba(77, 107, 254, 0.1); padding: 20px; border-radius: 8px; border: 1px solid rgba(77, 107, 254, 0.3); height: 100%;'>
        <h4 style='color: #4d6bfe; text-align: center;'>🔍 过程再现而非结果预测</h4>
        <p style='color: #c9d1d9; font-size: 0.95em; text-align: center;'>
        重点不在于预测精确的人口数字，而在于揭示特定历史节点上，决策者面临何种约束、如何思考、为何在彼时彼地做出特定选择。
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div style='background: rgba(0, 230, 118, 0.1); padding: 20px; border-radius: 8px; border: 1px solid rgba(0, 230, 118, 0.3); height: 100%;'>
        <h4 style='color: #00e676; text-align: center;'>🎮 策略探索而非最优求解</h4>
        <p style='color: #c9d1d9; font-size: 0.95em; text-align: center;'>
        通过调整智能体的认知风格（如"风险偏好""时间视野"），用户可以观察同一历史条件下不同决策逻辑如何导向不同的政策路径与长期后果。
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div style='background: rgba(255, 82, 82, 0.1); padding: 20px; border-radius: 8px; border: 1px solid rgba(255, 82, 82, 0.3); height: 100%;'>
        <h4 style='color: #ff5252; text-align: center;'>🌐 系统思维而非线性分析</h4>
        <p style='color: #c9d1d9; font-size: 0.95em; text-align: center;'>
        模型将经济、人口、社会、政治置于一个相互作用、延迟反馈的复杂系统中，展现局部优化可能导致长期失衡。
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 结语部分
    st.markdown("---")
    st.markdown("### 🌉 结语：作为理论与方法桥梁的Espark")
    
    st.markdown("""
    <div style='background: rgba(22, 27, 34, 0.6); border: 1px solid #30363d; border-radius: 12px; padding: 25px; margin-bottom: 20px;'>
    <p style='color: #e6edf3; font-size: 1.05em; line-height: 1.6; text-align: center;'>
    <strong>Espark Policy Lab</strong> 本质上是将间断均衡理论与复杂自适应系统理论<strong>操作化</strong>为可计算、可交互的生成式仿真模型。它既是对两大理论的一次实证检验与技术实现，也为公共政策研究提供了一种新的方法论工具——通过构建"认知可解释"的智能体，在虚拟实验室中复现政策系统的演化动力学，从而在历史分析与未来推演之间架起一座桥梁。这不仅有助于深化我们对政策变迁规律的理解，也为面向不确定未来的战略规划提供了可贵的"试错空间"与洞察来源。
    </p>
    </div>
    """, unsafe_allow_html=True)

# --- 场景 4：市场对标 (深度市场版) ---
elif menu == "🌐 市场对标 (Market)":
    st.markdown("# 🌐 市场对标分析")
    
    # 引入
    st.markdown("""
    <div style='background: rgba(22, 27, 34, 0.6); border: 1px solid #30363d; border-radius: 12px; padding: 25px; margin-bottom: 20px;'>
    <p style='color: #e6edf3; font-size: 1.05em; line-height: 1.6;'>
    Espark Policy Lab 处于<strong>传统政策模拟工具</strong>与<strong>生成式AI应用</strong>的交叉领域。相比传统ABM工具，我们增加了认知仿真维度；相比通用AI助手，我们聚焦于政策制定过程的专业模拟。
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 三个维度的对标
    tabs = st.tabs(["🔄 传统政策模拟", "🧠 认知AI平台", "🚀 新兴竞争者"])
    
    with tabs[0]:
        st.markdown("### 🔄 传统政策模拟工具对比")
        
        data_traditional = pd.DataFrame({
            "产品/平台": ["NetLogo", "AnyLogic", "PolicyEngine", "iDS (清华大学)", "GAMA Platform"],
            "类型": ["教育/研究ABM", "商业仿真", "税收福利微观模拟", "中国政策仿真系统", "地理空间ABM"],
            "核心方法": ["基于规则ABM", "多方法仿真", "微观模拟", "系统动力学+ABM", "地理ABM"],
            "认知能力": ["❌ 无", "❌ 无", "❌ 无", "⚠️ 有限", "❌ 无"],
            "中国政策适配": ["低", "低", "中(海外中国研究)", "高(本土开发)", "中"],
            "可解释性": ["中等(代码)", "中等(可视化)", "高(透明算法)", "中等", "中等"],
            "使用门槛": ["中(编程)", "高(建模)", "中(配置)", "高(专业)", "高(编程)"],
            "代表用户": ["高校教学", "企业咨询", "智库研究", "政府智库", "城市规划"]
        })
        
        # 高亮Espark的对比
        st.dataframe(
            data_traditional.style.apply(
                lambda x: ['background: rgba(77, 107, 254, 0.2)' if i == 3 else '' for i in range(len(x))], 
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("""
        <div style='background: rgba(77, 107, 254, 0.1); border-left: 4px solid #4d6bfe; padding: 15px; margin-top: 15px; border-radius: 0 8px 8px 0;'>
        <h4 style='color: #4d6bfe; margin-top: 0;'>Espark 的差异化优势</h4>
        <ul style='color: #c9d1d9;'>
        <li><strong>认知维度突破</strong>：传统工具只能模拟"行为"，Espark模拟"思考过程"</li>
        <li><strong>降低使用门槛</strong>：无需编程，通过自然语言Prompt调整模型</li>
        <li><strong>中国语境深度适配</strong>：理解"民主集中制"、"五年规划"等中国特色概念</li>
        <li><strong>可解释性革命</strong>：提供完整思维链，而不仅是输入输出</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 🧠 认知AI平台对比")
        
        data_ai = pd.DataFrame({
            "产品/平台": ["ChatGPT + 插件", "Claude Projects", "GPTs (OpenAI)", "DeepSeek", "文心一言"],
            "定位": ["通用AI助手", "企业级AI项目", "自定义AI助手", "通用大模型", "中文大模型"],
            "政策分析能力": ["中(需引导)", "中高(可定制)", "中(依赖Prompt)", "中", "中高(中文理解)"],
            "模拟仿真功能": ["❌ 无内置", "❌ 无内置", "❌ 无内置", "❌ 无内置", "❌ 无内置"],
            "时间维度": ["无记忆", "项目记忆", "有限上下文", "128K上下文", "有限上下文"],
            "决策过程展示": ["思考链(需要求)", "思考链", "思考链", "思考链", "思考链"],
            "政策专业度": ["依赖Prompt工程", "可专业化", "依赖Prompt工程", "依赖Prompt", "对中文政策较好"],
            "适合场景": ["政策问答", "政策文档分析", "简单政策咨询", "技术性政策分析", "中文政策理解"]
        })
        
        st.dataframe(
            data_ai.style.apply(
                lambda x: ['background: rgba(0, 230, 118, 0.2)' if i == 3 else '' for i in range(len(x))], 
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("""
        <div style='background: rgba(0, 230, 118, 0.1); border-left: 4px solid #00e676; padding: 15px; margin-top: 15px; border-radius: 0 8px 8px 0;'>
        <h4 style='color: #00e676; margin-top: 0;'>Espark 的专业化优势</h4>
        <ul style='color: #c9d1d9;'>
        <li><strong>领域专业化</strong>：不是通用对话，而是针对政策模拟的深度定制</li>
        <li><strong>仿真系统内置</strong>：完整的ABM框架+时间序列模拟，非单次问答</li>
        <li><strong>多轮决策记忆</strong>：完整的政策演进历史，而非独立对话</li>
        <li><strong>结构化输出</strong>：生成标准的JSON决策记录，便于分析</li>
        <li><strong>可视化集成</strong>：内置图表展示政策演进轨迹</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 🚀 新兴竞争者与替代方案")
        
        data_emerging = pd.DataFrame({
            "项目/平台": ["Stanford Smallville", "Microsoft Autogen", "Constitutional AI", "决策智能平台", "数字孪生城市"],
            "类型": ["生成式智能体社会", "多智能体框架", "价值观对齐AI", "企业决策支持", "城市级仿真"],
            "相似度": ["高(方法论)", "中(多智能体)", "低(价值观)", "中(决策支持)", "低(尺度不同)"],
            "发展阶段": ["学术研究", "开源框架", "研究阶段", "商业应用", "政府项目"],
            "开源状态": ["开源", "开源", "部分开源", "闭源", "闭源"],
            "政策聚焦": ["社会交互", "任务协作", "AI安全", "商业决策", "城市治理"],
            "中国适应性": ["低", "中", "低", "中", "高(本土开发)"],
            "威胁级别": ["高(学术领先)", "中(技术框架)", "低", "中(商业竞争)", "低(不同领域)"]
        })
        
        st.dataframe(
            data_emerging.style.apply(
                lambda x: ['background: rgba(255, 82, 82, 0.2)' if i == 7 else '' for i in range(len(x))], 
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("""
        <div style='background: rgba(255, 82, 82, 0.1); border-left: 4px solid #ff5252; padding: 15px; margin-top: 15px; border-radius: 0 8px 8px 0;'>
        <h4 style='color: #ff5252; margin-top: 0;'>Espark 的护城河</h4>
        <ul style='color: #c9d1d9;'>
        <li><strong>领域聚焦</strong>：专注公共政策，特别是中国政策语境</li>
        <li><strong>理论深度</strong>：基于间断均衡、复杂自适应等成熟理论</li>
        <li><strong>用户体验</strong>：Streamlit实现零配置、交互式体验</li>
        <li><strong>快速迭代</strong>：基于开源生态，快速响应需求</li>
        <li><strong>数据隐私</strong>：可完全本地部署，保护敏感政策数据</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # SWOT分析
    st.markdown("---")
    st.markdown("### 📊 Espark SWOT分析")
    
    swot_cols = st.columns(4)
    
    with swot_cols[0]:
        st.markdown("""
        <div style='background: rgba(0, 230, 118, 0.15); padding: 20px; border-radius: 8px; border: 1px solid #00e676; height: 100%;'>
        <h4 style='color: #00e676; text-align: center;'>👍 优势 (Strengths)</h4>
        <ul style='color: #c9d1d9; font-size: 0.9em;'>
        <li>生成式智能体的认知仿真能力</li>
        <li>中国政策语境的深度理解</li>
        <li>零代码交互体验</li>
        <li>完整的思维链可解释性</li>
        <li>基于成熟理论框架</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with swot_cols[1]:
        st.markdown("""
        <div style='background: rgba(255, 82, 82, 0.15); padding: 20px; border-radius: 8px; border: 1px solid #ff5252; height: 100%;'>
        <h4 style='color: #ff5252; text-align: center;'>👎 劣势 (Weaknesses)</h4>
        <ul style='color: #c9d1d9; font-size: 0.9em;'>
        <li>依赖大模型API（成本/稳定性）</li>
        <li>模拟规模有限（单智能体）</li>
        <li>缺乏真实历史数据验证</li>
        <li>用户群体小众（政策研究者）</li>
        <li>计算性能受Streamlit限制</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with swot_cols[2]:
        st.markdown("""
        <div style='background: rgba(255, 183, 77, 0.15); padding: 20px; border-radius: 8px; border: 1px solid #ffb74d; height: 100%;'>
        <h4 style='color: #ffb74d; text-align: center;'>🚀 机遇 (Opportunities)</h4>
        <ul style='color: #c9d1d9; font-size: 0.9em;'>
        <li>政府数字化转型需求</li>
        <li>AI for Science政策支持</li>
        <li>高校计算社会科学教学需求</li>
        <li>智库研究工具升级</li>
        <li>海外中国研究市场</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with swot_cols[3]:
        st.markdown("""
        <div style='background: rgba(77, 107, 254, 0.15); padding: 20px; border-radius: 8px; border: 1px solid #4d6bfe; height: 100%;'>
        <h4 style='color: #4d6bfe; text-align: center;'>⚠️ 威胁 (Threats)</h4>
        <ul style='color: #c9d1d9; font-size: 0.9em;'>
        <li>大厂进入政策AI领域</li>
        <li>技术路线快速迭代</li>
        <li>政策敏感性带来的合规风险</li>
        <li>开源竞品的同质化</li>
        <li>用户习惯难以改变</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 市场定位图
    st.markdown("---")
    st.markdown("### 🗺️ 市场定位图谱")
    
    # 创建一个简单的市场定位图表
    fig = go.Figure()
    
    # 各产品在二维空间的位置
    products = {
        "NetLogo": (2, 8, "传统ABM"),
        "AnyLogic": (3, 7, "商业仿真"),
        "PolicyEngine": (5, 6, "微观模拟"),
        "ChatGPT": (8, 4, "通用AI"),
        "Claude": (7, 5, "企业AI"),
        "Smallville": (9, 9, "生成式智能体"),
        "Autogen": (8, 7, "多智能体"),
        "Espark": (7, 9, "政策G-ABM")
    }
    
    for product, (x, y, category) in products.items():
        color = "#4d6bfe" if product == "Espark" else "#666"
        size = 20 if product == "Espark" else 12
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=size, color=color),
            text=[product],
            textposition="top center",
            name=category,
            hoverinfo='text',
            hovertext=f"{product}: {category}"
        ))
    
    fig.update_layout(
        title="市场定位：传统性 vs AI驱动性",
        xaxis_title="AI驱动性 (低 → 高)",
        yaxis_title="政策专业性 (低 → 高)",
        template="plotly_dark",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 10], showgrid=True, gridcolor='#333'),
        yaxis=dict(range=[0, 10], showgrid=True, gridcolor='#333'),
        showlegend=False
    )
    
    # 添加象限说明
    fig.add_annotation(x=2.5, y=2.5, text="传统工具区", showarrow=False, font=dict(color="#888", size=12))
    fig.add_annotation(x=7.5, y=2.5, text="通用AI区", showarrow=False, font=dict(color="#888", size=12))
    fig.add_annotation(x=2.5, y=7.5, text="专业仿真区", showarrow=False, font=dict(color="#888", size=12))
    fig.add_annotation(x=7.5, y=7.5, text="前沿创新区", showarrow=False, font=dict(color="#4d6bfe", size=14, weight="bold"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 总结
    st.markdown("---")
    st.markdown("### 🎯 总结：Espark的独特价值主张")
    
    st.markdown("""
    <div style='background: rgba(22, 27, 34, 0.6); border: 1px solid #30363d; border-radius: 12px; padding: 25px; margin-bottom: 20px;'>
    <p style='color: #e6edf3; font-size: 1.05em; line-height: 1.6;'>
    <strong>Espark Policy Lab</strong> 填补了市场空白：在<strong>传统政策模拟工具</strong>（如NetLogo、AnyLogic）与<strong>通用AI助手</strong>（如ChatGPT）之间，提供了一个专门针对公共政策制定过程的<strong>认知仿真平台</strong>。
    </p>
    
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;'>
    <div style='background: rgba(77, 107, 254, 0.1); padding: 15px; border-radius: 8px;'>
    <h5 style='color: #4d6bfe; margin-top: 0;'>相比传统政策模拟工具：</h5>
    <ul style='color: #c9d1d9; font-size: 0.9em;'>
    <li>✓ 增加了决策者的认知维度</li>
    <li>✓ 大幅降低了使用门槛</li>
    <li>✓ 提供可解释的思维链</li>
    <li>✓ 更好地理解中国政策语境</li>
    </ul>
    </div>
    
    <div style='background: rgba(0, 230, 118, 0.1); padding: 15px; border-radius: 8px;'>
    <h5 style='color: #00e676; margin-top: 0;'>相比通用AI助手：</h5>
    <ul style='color: #c9d1d9; font-size: 0.9em;'>
    <li>✓ 内置完整的政策仿真框架</li>
    <li>✓ 支持多轮决策和历史回溯</li>
    <li>✓ 专门的政策分析工作流</li>
    <li>✓ 集成可视化与数据导出</li>
    </ul>
    </div>
    </div>
    
    <p style='color: #e6edf3; font-size: 1.05em; line-height: 1.6; margin-top: 20px;'>
    <strong>目标用户：</strong> 高校公共政策/政治学研究者、政府智库分析师、计算社会科学学生、对政策制定过程感兴趣的公众。
    </p>
    
    <p style='color: #e6edf3; font-size: 1.05em; line-height: 1.6;'>
    <strong>核心价值：</strong> 不是替代传统ABM或通用AI，而是在两者之间创造新的工具类别——<strong>认知政策仿真器</strong>，让政策分析从"计算社会"走向"认知社会"。
    </p>
    </div>
    """, unsafe_allow_html=True)

# --- 场景 5：智能体科普 (About) - 【保留原模版内容】 ---
elif menu == "📚 智能体科普 (About)":
    st.markdown("# 📚 什么是生成式智能体 (Generative Agents)?")
    st.markdown("### 从“计算社会科学”到“生成式社会科学”的范式转移")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#ff5252">🚫 传统 ABM (Rule-based)</h4>
            <p>基于固定规则的“物理仿真”。</p>
            <ul>
                <li><b>Agent 本质：</b> 冷冰冰的数学公式。</li>
                <li><b>决策逻辑：</b> if 压力 > 50 then 改变。</li>
                <li><b>局限性：</b> 无法模拟复杂的政治权衡、犹豫和模糊性。</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#00e676">✅ Espark G-ABM (Cognitive)</h4>
            <p>基于 LLM 的“认知仿真”。</p>
            <ul>
                <li><b>Agent 本质：</b> 拥有记忆、会反思的数字决策者。</li>
                <li><b>决策逻辑：</b> 基于 Prompt 的推理链 (Chain of Thought)。</li>
                <li><b>优势：</b> 能理解“民主集中制”、“跨代际责任”等复杂概念。</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🧩 本模型的核心认知架构")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. 感知 (Perception)**\n\n能够读取宏观经济数据和 T-20 年的劳动力滞后反馈。")
    c2.markdown("**2. 记忆 (Memory)**\n\n记住上一轮的政策效果（反馈），形成路径依赖。")
    c3.markdown("**3. 决策 (Action)**\n\n在“经济增长”与“人口安全”的注意力竞争中做出权衡。")