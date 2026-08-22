from langchain_ollama import ChatOllama

from utils.DatasetManager import DatasetManager
from utils.analyzer import NetCDFAnalyzer1
from agents.process_agent import ProcessAgent
from agents.coding_agent import CodingAgent
from test import MainAgent


def main():
    # 1. 初始化 DatasetManager
    manager = DatasetManager()

    # 2. 初始化 LLM
    llm = ChatOllama(model="qwen3-coder:480b-cloud", temperature=0)

    # 3. 初始化 Coding Agent
    coding_agent = CodingAgent(llm=llm, max_attempts=5, verbose=True)

    # 4. 初始化 Process Agent
    process_agent = ProcessAgent(
       #coding_agent=coding_agent,
        model="qwen3-coder:480b-cloud",
        #max_steps=10,
        verbose=True,
        manager=manager,
    )

    # 5. 初始化 Data Analyzer
    analyzer = NetCDFAnalyzer1()

    # 6. 初始化 Main Agent
    main_agent = MainAgent(
        dataset_manager=manager,
        process_agent=process_agent,
        data_analyzer=analyzer,
        llm=llm
    )

    # 7. 运行交互
    print("=" * 60)
    print("NetCDF 多智能体系统已启动")
    print("输入 'quit' 退出")
    print("=" * 60)
    import uuid
    session_id = str(uuid.uuid4()) # 在循环外生成唯一的 session_id
    print(f"当前会话 ID: {session_id}")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue

        response = main_agent.run(user_input, session_id=session_id)
        print("\n" + "=" * 60)
        print("最终报告:")
        print(response)
        print("=" * 60)


if __name__ == "__main__":
    main()