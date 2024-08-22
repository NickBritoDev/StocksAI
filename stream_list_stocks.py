#IMPORT DA LIBS
import os
import yfinance as yf
import streamlit as st
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI

# LLM OPENAI - GPT 3.5 TURBO
llm = ChatOpenAI(model="gpt-3.5-turbo")

# YAHOO FINANCE TOOL 
def fetch_stock_price(ticket):
    stocks = yf.download(ticket, start="2023-08-08", end="2024-01-08")
    return stocks

yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stock prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)


# AGENTE DE BUSCA DE MERCADO
stock_price_analyst = Agent(
    role="Senior stock price analyst",
    goal="Find the {ticket} stock price and analyze trends",
    backstory="You're highly experienced in analyzing the price of a specific stock and making predictions about its future price.",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=False,
    tools=[yahoo_finance_tool]
)

# TAREFA
get_stock_price = Task(
    description="Analyze the stock {ticket} price history and create a trend analysis of up, down, or sideways",
    expected_output="Specify the current trend of the stock price - up, down, or sideways. Eg. stock= 'APPL', price UP",
    agent=stock_price_analyst
)

# DUCK DUCK GO SEARCH TOOL
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

# AGENTE DE BUSCA DE NOTÍCIAS
news_analyst = Agent(
    role="Stock news analyst",
    goal="Create a short summary of the news related to the stock {ticket} company. Specify the current trend - up, down, or sideways with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.",
    backstory="You're highly experienced in analyzing market trends and news and have tracked assets for more than 10 years. You're also a master-level analyst in traditional markets with a deep understanding of human psychology. You understand news, their titles, and information, but you view them with a healthy dose of skepticism. You also consider the source of the news articles.",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=False,
    tools=[search_tool]
)

# TAREFA
get_news = Task(
    description=f"Take the stock and always include BTC to it (if not requested). Use the search tool for each one individually. The current date is {datetime.now()}. Compose the result into a helpful report.",
    expected_output="""A summary of the overall market and one sentence summary for each requested asset. Include a fear/greed score for each asset based on the news.
    Use format: 
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent=news_analyst
)

# AGENTE DE ANÁLISE DE AÇÕES
stock_analyst_write = Agent(
    role="Senior stock analyst writer",
    goal="Analyze the trend price and news and write an insightful, compelling, and informative 3-paragraph-long newsletter based on the stock report and price trend.",
    backstory="You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate with wider audiences. You understand macro factors and combine multiple theories - e.g., cycle theory and fundamental analysis. You're able to hold multiple opinions when analyzing.",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
    tools=[search_tool]
)

# TAREFA
write_analyses = Task(
    description="Use the stock price trend and the stock news report to create an analysis and write the newsletter about the company that is brief and highlights the most important points. Focus on the stock price trend, news, and fear/greed score. What are the near future considerations? Include the previous analysis of stock trend and news summary.",
    expected_output="""An eloquent 3-paragraph newsletter formatted as markdown in an easy-to-read manner. It should contain:
    - 3 bullet points executive summary
    - Introduction - set the overall picture and spike up the interest
    - Main part provides the meat of the analysis including the news summary and fear/greed scores
    - Summary - key facts and concrete future trend prediction - up, down, or sideways.
    """,
    agent=stock_analyst_write,
    context=[get_stock_price, get_news]
)

# CONFIGURAÇÃO DO CREW
crew = Crew(
    agents=[stock_price_analyst, news_analyst, stock_analyst_write],
    tasks=[get_stock_price, get_news, write_analyses],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of research:")
        st.write(results['final_output'])