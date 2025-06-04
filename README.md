
## 一、网页核心信息  
### 1.1 基础信息  
- **网页类型**：技术文档（Ragas工具使用指南）  
- **标题**：LangGraph - Ragas  
- **核心主题**：通过LangGraph构建ReAct代理，并使用Ragas评估其性能  


## 二、Ragas核心功能模块  
### 2.1 评估指标体系  
#### 2.1.1 代理/工具使用场景指标  
- **Tool Call Accuracy**：评估LLM是否正确调用工具及传入参数（二进制指标，1为正确，0为错误）  
- **Agent Goal Accuracy**：评估代理是否达成用户目标（二进制指标，1为达成，0为未达成）  

#### 2.1.2 其他核心指标分类  
- **检索增强生成（RAG）**：Context Precision、Context Recall、Response Relevancy、Faithfulness  
- **自然语言比较**：Factual Correctness、Semantic Similarity、BLEU Score、ROUGE Score  
- **通用指标**：Aspect Critic、Rubrics Based Scoring  


## 三、ReAct代理构建与评估教程（核心内容）  
### 3.1 前置条件  
- **环境要求**：Python 3.8+，需理解LangGraph、LangChain、LLM基础概念  
- **依赖安装**：  
  ```bash  
  %pip install langgraph==0.2.44  
  %pip install ragas  
  %pip install nltk  
  ```  

### 3.2 外部组件初始化  
#### 3.2.1 模拟API响应（可选）  
```python  
metal_price = {  
    "gold": 88.1553,  
    "silver": 1.0523,  
    "platinum": 32.169,  
    # 完整JSON数据见网页代码（包含30+金属价格键值对）  
}  
```  

### 3.3 定义工具函数  
```python  
from langchain_core.tools import tool  

@tool  
def get_metal_price(metal_name: str) -> float:  
    """获取指定金属的当前价格（美元/克）"""  
    try:  
        metal_name = metal_name.lower().strip()  
        if metal_name not in metal_price:  
            raise KeyError(f"金属{metal_name}未找到")  
        return metal_price[metal_name]  
    except Exception as e:  
        raise Exception(f"获取价格失败：{str(e)}")  
```  

### 3.4 绑定LLM与工具  
```python  
from langchain_openai import ChatOpenAI  

tools = [get_metal_price]  
llm = ChatOpenAI(model="gpt-4o-mini")  
llm_with_tools = llm.bind_tools(tools)  # 将工具绑定到LLM  
```  

### 3.5 定义状态与对话逻辑  
#### 3.5.1 状态类（追踪对话消息）  
```python  
from langgraph.graph import END  
from langchain_core.messages import AnyMessage  
from langgraph.graph.message import add_messages  
from typing import Annotated  
from typing_extensions import TypedDict  

class GraphState(TypedDict):  
    messages: Annotated[list[AnyMessage], add_messages]  # 存储对话消息列表  
```  

#### 3.5.2 对话终止条件函数  
```python  
def should_continue(state: GraphState):  
    """根据最后一条消息是否包含工具调用，决定是否继续对话"""  
    last_message = state["messages"][-1]  
    return "tools" if last_message.tool_calls else END  
```  

### 3.6 构建状态图（StateGraph）  
#### 3.6.1 节点定义  
- **助手节点**：生成LLM响应  
  ```python  
  def assistant(state: GraphState):  
      response = llm_with_tools.invoke(state["messages"])  
      return {"messages": [response]}  
  ```  
- **工具节点**：调用外部工具  
  ```python  
  from langgraph.prebuilt import ToolNode  
  tool_node = ToolNode(tools)  # 初始化工具节点  
  ```  

#### 3.6.2 图结构构建  
```python  
from langgraph.graph import START, StateGraph  

builder = StateGraph(GraphState)  
builder.add_node("assistant", assistant)  # 添加助手节点  
builder.add_node("tools", tool_node)  # 添加工具节点  
builder.add_edge(START, "assistant")  # 入口节点为助手节点  
builder.add_conditional_edges("assistant", should_continue, ["tools", END])  # 条件边：根据工具调用决定流向  
builder.add_edge("tools", "assistant")  # 工具调用后回到助手节点  
react_graph = builder.compile()  # 编译图  
```  

### 3.7 消息格式转换（LangChain → Ragas）  
```python  
from ragas.integrations.langgraph import convert_to_ragas_messages  

# LangChain消息列表（包含HumanMessage、AIMessage、ToolMessage）  
messages = [HumanMessage(content="What is the price of copper?")]  
result = react_graph.invoke({"messages": messages})  
ragas_trace = convert_to_ragas_messages(result["messages"])  # 转换为Ragas格式  
```  

### 3.8 性能评估代码  
#### 3.8.1 工具调用准确性评估  
```python  
from ragas.metrics import ToolCallAccuracy  
from ragas.dataset_schema import MultiTurnSample  
from ragas.messages import ToolCall  

sample = MultiTurnSample(  
    user_input=ragas_trace,  
    reference_tool_calls=[ToolCall(name="get_metal_price", args={"metal_name": "copper"})]  
)  
tool_accuracy_scorer = ToolCallAccuracy()  
await tool_accuracy_scorer.multi_turn_ascore(sample)  # 输出：1.0  
```  

#### 3.8.2 代理目标达成率评估  
```python  
from ragas.metrics import AgentGoalAccuracyWithReference  
from ragas.llms import LangchainLLMWrapper  

# 测试用户查询“10克银价”  
messages = [HumanMessage(content="What is the price of 10 grams of silver?")]  
result = react_graph.invoke({"messages": messages})  
ragas_trace = convert_to_ragas_messages(result["messages"])  

sample = MultiTurnSample(  
    user_input=ragas_trace,  
    reference="Price of 10 grams of silver"  # 目标描述  
)  
scorer = AgentGoalAccuracyWithReference()  
scorer.llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))  
await scorer.multi_turn_ascore(sample)  # 输出：1.0  
```  


## 四、集成能力与工具支持  
### 4.1 兼容框架  
- **LangChain**：用于定义工具（`@tool`装饰器）和LLM交互  
- **LangGraph**：构建代理工作流图（StateGraph），支持可视化流程（`draw_mermaid_png()`）  
- **其他集成**：Amazon Bedrock、Haystack、LlamaIndex、LangSmith等  

### 4.2 数据格式转换  
- 提供专用函数`convert_to_ragas_messages()`，将LangChain消息（包含ToolCall、AIMessage等）转换为Ragas评估所需格式，确保指标计算兼容性  


## 五、代码执行流程总结  
1. **代理初始化**：定义工具→绑定LLM→构建状态图  
2. **对话执行**：用户输入→助手节点生成含工具调用的响应→工具节点获取数据→循环直至无工具调用  
3. **结果评估**：转换消息格式→调用Ragas指标类（如`ToolCallAccuracy`）→计算得分  


## 六、版本与社区支持  
### 6.1 版本信息  
- 当前稳定版本：v0.2.15  
- 提供版本迁移指南（如从v0.1到v0.2）  

### 6.2 技术支持  
- **办公时间**：通过网页链接报名一对一配置评估帮助  
- **文档资源**：包含快速入门、核心概念、自定义指标开发等详细指南