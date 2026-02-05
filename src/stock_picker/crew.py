from asyncio import tasks
from multiprocessing import process
from tabnanny import verbose
from crew.stock_picker.src.stock_picker.tools import push_tool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel, Field, config
from crewai_tools import SerperDevTool

class TrendingCompany(BaseModel):
    """
    A company that is in news and attracting attention
    """
    name: str = Field(description="Company Name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reson this comapny is trending in the news")
    
class TrendingCompanyList(BaseModel):
    """
    List of multiple trending companies that are in the news 
    """
    companies: List[TrendingCompany] = Field(description="List of companies trending in the news")
    
class TrendingCompanyResearch(BaseModel):
    """
    Detailed research on a company
    """
    name: str = Field(description="Company Name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability for investmant")
    
class TrendingCompanyResearchList(BaseModel):
    """
    List of deatiled researched companies
    """
    companies_research: List[TrendingCompany] = Field(description="deatiled list of tranding companies research")

@CrewBase
class StockPicker():
    """StockPicker crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    
    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(config=self.agents_config['trending_company_finder'],
                     tools=[SerperDevTool()])
        
    @agent
    def financial_researcher(self) -> Agent:
        return Agent(config=self.agents_config['financial_researcher'],
                     tools=[SerperDevTool()])
        
    @agent
    def stock_picker(self) -> Agent:
        return Agent(config=self.agents_config['stock_picker'],
                     tools=[push_tool()])
    
    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['find_trending_companies'],
            output_pydantic=TrendingCompanyList,
        )
        
    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['research_trending_companies'],
            output_pydantic=TrendingCompanyResearchList,
        )
        
    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.tasks_config['pick_best_company']
        )
        
    @crew
    def crew(self) -> Crew:
        """
        Create the Stockpicker crew
        """
        
        manager = Agent(
            config=self.agents_config['manager'],
            allow_delegation=True
        )
        
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager        
        )