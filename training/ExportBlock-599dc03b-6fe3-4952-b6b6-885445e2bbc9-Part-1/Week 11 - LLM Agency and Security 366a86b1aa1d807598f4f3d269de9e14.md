# Week 11 - LLM Agency and Security

# 1. Reasoning vs Agency

> **Reasoning -** process information and derive conclusions internally (deductive, inductive, abductive, formal analogical, commonsense). Used for factual Q&A, puzzles, summarisation, IE.
> 

> **Agency -** capacity to act autonomously and purposefully on the world. Takes external inputs, produces actions.
> 

## 1.1 How LLMs make plans

When asked to act the LLM doesn’t just answer it produces a plan. It can be implicit like model emits an action token or eexplicit *chain of though style First search the web, then summarise, then email. Plans are intermediate outputs that re-enter the system to drive decisions. 

## 1.2 Function calling

Before MCP (Model-Context Protocol) OpenAI introduced function calling: the LLM was given a JSON schema describing available tools and could emit a structure request like funciton: get weather. 

The host then executed the function and few the result back. This has no interoperabilitiy. 

## 1.3 The ReACT pattern

ReACT (reasoning + acting) is the dominant pattern for orchestrating LLM agents. The model alternates between **Thought (reasoning step), Action (tool call) and Observation (tool result).** 

# 2. Model-Context Protocol (MCP)

Anthropic introduced MCP in 2024 as a universal mechanism for LLMs to access external services. Before MCP every LLM provider had a bespoke way to wire in tools. MCP standardises the protocol so any client can talk to any server.

> **MCP (Model Context Protocol)** a standardised protocol for 2 way connection between an LLM and external tools. Three components: Host Client and server.
> 

## 2.1 Three components

- **MCP Host -** the wrapping environment containing the LLM and Clients. e.g. Claude Desktop, Cursor IDE.
- **MCP Client -** one per tool/server. Lives inside the Host. Translates between LLM intent and server protocol.
- **MCP Server** - external tool itself (web search, git actions, email file system access). Lives off the platform. Each tool ships its own server.

## 2.2 The sequence flow.

![diagram.jpg](Week%2011%20-%20LLM%20Agency%20and%20Security/diagram.jpg)

**Resources vs tools.** MCP distinguishes between two served exposed capabilities:

- **Tools -** functions LLM can invoke (web search, send email, run query)
- **Resources** - data the LLM can read (file contents, document collections, knowledge entries).

<aside>
💡

Note there is only always 1 host. client per action and server per client. 

</aside>

## 2.3 MCP advantages

- **Scalability.** Servers exist off-platform, so each tool can be scaled independently. If web search is used more than email, scale just the web search without touching the rest.
- **Modularity.** Reuse pre-built MCP components across projects. New apps don’t re-implement integrations.

Other practical advantages worth knowing:

- **Vendor neutrality -** any MCP Host can talk to any MCP server so you are not looked into one LLM provider’s tool ecosystem.
- **Local execution -** servers can run on the user’s machine, keeping sensitive data off external APIs.

# 3. LangChain vs LangGraph

> **LangChain -** framework for sequential workflows combining NLP components (Retrieve → prompt → LLM).
> 

> **LangGraph -** framework for non-linear, graph based workflows combining multiple agents/components with shared state. Excels at multi0agent orchestration.
> 

**Scenario:**

Research assistant breaking down questions into sub-queries → LangGraph branches these queries.

**LangGraph state.** A langGraph workflow has a shared state object often typed as a dictionary that nodes read and update. Standard nodes include Process, Summarise, AddTasks, CompleteTasks - each modifies the state and routes to the next node based on the state’s contents.