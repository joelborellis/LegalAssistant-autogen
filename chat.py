from autogen import UserProxyAgent, ConversableAgent, config_list_from_json, GroupChat, GroupChatManager
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

# retrieve some variables from .env
ASSISTANT_ID = os.environ.get("ASSISTANT_ID", None) # id of the Assistant that we want to retrieve
OPENAI_KEY = os.environ.get("OPENAI_KEY", None)

# create client for OpenAI
client = OpenAI(api_key=OPENAI_KEY)

###     file operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

# function to generate the legal notes
def create_notes(case):
    print(case)
    ## load case
    text = open_file('./data/txt/nyt-v-openai-microsoft.txt').replace('\n\n', '\n')
    return text

def main():

    # Load LLM inference endpoints from an env variable or a file
    # See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
    # and OAI_CONFIG_LIST_sample.
    # For example, if you have created a OAI_CONFIG_LIST file in the current working directory, that file will be used.
    config_list_gpt4 = config_list_from_json(
            "OAI_CONFIG_LIST",
            filter_dict={
                "model": ["gpt-4-0125-preview", "gpt-3.3-turbo", "gpt-3.5"],
            },
    )

    # Retrieve an existing assistant already setup as an OpenAI Assistant
    # this is OpenAI Assistant stuff
    legalnotes_assistant = client.beta.assistants.retrieve(
                        assistant_id=ASSISTANT_ID,
                        ) 
    
    legalcourtroom_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_WrhyKUEYgpfH7PLkyzynuEeD",
                        ) 
    
    planner_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_aNdSXqTtqv9wGcOBsJyTaLXi",
                        ) 

    legalnotes_config = {
            "config_list": config_list_gpt4,
            "assistant_id": legalnotes_assistant.id,
            "tools": [
                {
                    "type": "function",
                    "function": create_notes,
                }
                    ]
        }
    
    legalcourtroom_config = {
            "config_list": config_list_gpt4,
            "assistant_id": legalcourtroom_assistant.id,
        }
    
    planner_assistant_config = {
            "config_list": config_list_gpt4,
            "assistant_id": planner_assistant.id,
        }

    # this is autogen stuff defining the agent that is going to be in the group
    legalnotes_agent = GPTAssistantAgent(
            name="LegalAssistant",
            instructions=None,
            llm_config=legalnotes_config,
        )

    legalnotes_agent.register_function(
            function_map={
                "create_notes": create_notes,
            }
        )
    
    # this is autogen stuff defining the agent that is going to be in the group
    legalcourtroom_agent = GPTAssistantAgent(
            name="LegalCourtroomAssistant",
            instructions=None,
            llm_config=legalcourtroom_config,
        )

    # this is autogen stuff defining the agent that is going to be in the group
    planner_agent = GPTAssistantAgent(
            name="LegalAssistantPlanner",
            instructions=None,
            llm_config=planner_assistant_config,
        )

    user_proxy = UserProxyAgent(
            name="Admin",
            system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this Admin.",
            code_execution_config=False,
            max_consecutive_auto_reply=10,
            human_input_mode="ALWAYS",
            llm_config=False,
        )
    

    groupchat = GroupChat(agents=[user_proxy, planner_agent, legalnotes_agent, legalcourtroom_agent], messages=[], max_round=30, speaker_selection_method='round_robin')
    manager = GroupChatManager(groupchat=groupchat, name="legalassistant_manager", llm_config=False)

    print("initiating chat")

    # Let the assistant start the conversation.  It will end when the user types exit.

    user_proxy.initiate_chat(
            manager,
            message="""
            Write me legal notes for the case nyt-v-openai-microsoft.
            Write an opening statement based on the legal notes.
            """,
            silent=False
        )


if __name__ == "__main__":
    main()