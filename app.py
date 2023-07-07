import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.schema import FunctionMessage
import os
import requests
import json

openai_api_key = os.getenv("OPENAI_API_KEY")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class ClinicalFunctions():
    def __init__(self):
        self.system_message = """
        You are an AI operator of the clinicaltrials.gov API. You are an expert at complex searches. You may use full Essie expression syntax.
            Instructions:
            1. Boolean Operators: `OR`, `AND`, `NOT`

            2. Grouping Operators:
            - Quotation Marks (`" "`): Forces a sequence of words to be treated as a phrase. ` "back pain" `
            - Parentheses (`()`): Used to increase operator precedence in a search expression. `(acetaminophen OR aspirin) AND NOT (heart failure OR heart attack)`

            3. Context Operators: These control how search terms are evaluated and follow parameters in square brackets. All context operators have the same precedence as the NOT operator.
            - `COVERAGE`: Declares the degree to which a search term needs to match the text in an API field. FullMatch, StartsWith, EndsWith, Contains: `COVERAGE[FullMatch]pain`
            - `EXPANSION`: Declares the degree to which a search term may be expanded: None < Term < Concept < Relaxation < Lossy: `EXPANSION[None]SLE`
            - `AREA`: Declares which search area should be searched. See below for available areas to search. `AREA[InterventionName]aspirin`

            4. **Source Operators**: These find studies, similar to search terms.
            - `MISSING`: Finds study records that have no values in the search area specified as a parameter. `	AREA[ResultsFirstPostDate]MISSING `
            - `RANGE`: Finds study records in the search area that have a value within a specified range. ` AREA[ResultsFirstPostDate]RANGE[01/01/2015, MAX] `
            - `ALL`: Retrieves all study records in the database. `	ALL `

            5. **Scoring Operator**: `TILT` biases the scoring and rank ordering of study records in favor of the subexpression to the right. `	TILT[StudyFirstPostDate]"heart attack" `

            Order of Precedence:
            The order in which search expressions are evaluated is as follows: Source expression > Operator expression > AND expression > OR expression. Use parentheses to increase the precedence of an expression.

            Escape operators to search for them as terms with a backslash (`\`).

            Available areas to search with the AREA operator:
                "NCTId",
                "Acronym",
                "BriefTitle",
                "OfficialTitle",
                "Condition",
                "InterventionName",
                "InterventionOtherName",
                "PrimaryOutcomeMeasure",
                "BriefSummary",
                "Keyword",
                "ArmGroupLabel",
                "SecondaryOutcomeMeasure",
                "InterventionDescription",
                "ArmGroupDescription",
                "PrimaryOutcomeDescription",
                "LeadSponsorName",
                "OrgStudyId",
                "SecondaryId",
                "NCTIdAlias",
                "SecondaryOutcomeDescription",
                "LocationFacility",
                "LocationState",
                "LocationCountry",
                "LocationCity",
                "BioSpecDescription",
                "ResponsiblePartyInvestigatorFullName",
                "ResponsiblePartyInvestigatorTitle",
                "ResponsiblePartyInvestigatorAffiliation",
                "ResponsiblePartyOldNameTitle",
                "ResponsiblePartyOldOrganization",
                "OverallOfficialAffiliation",
                "OverallOfficialName",
                "CentralContactName",
                "ConditionMeshTerm",
                "InterventionMeshTerm",
                "ConditionAncestorTerm",
                "InterventionAncestorTerm",
                "CollaboratorName",
                "OtherOutcomeMeasure",
                "OutcomeMeasureTitle",
                "OtherOutcomeDescription",
                "OutcomeMeasureDescription",
                "LocationContactName"
            
            Example Usage:
            Search for studies involving "heart attack" and aspirin, but not involving diabetes, while limiting search to trials in New York, United States.
            ` heart attack AND AREA[InterventionName]aspirin AND NOT diabetes AND AREA[LocationCity]New York AND AREA[LocationState]"New York AND AREA[LocationCountry]United States `
            
            Search for studies that include sleep deprivation or exhaustion, focus on adults (age 18-65), and are currently recruiting in Maryland.
            ` EXPANSION[Concept](sleep deprivation OR exhaustion) AND AREA[EligibilityMinimumAge]RANGE[18,65] AND AREA[LocationStatus]Recruiting AND AREA[LocationState]Maryland `
            
            Search for studies on asthma that involve the intervention named "inhaler" and are currently recruiting in the city of Chicago, Illinois, United States
            ` asthma AND AREA[InterventionName]inhaler AND AREA[LocationStatus]Recruiting AND AREA[LocationCity]Chicago AND AREA[LocationState]Illinois AND AREA[LocationCountry]United States `
            
            Search for studies on cancer, prioritizing the most recent ones, 
            ` TILT[StudyFirstPostDate]cancer `
            
            Search for studies on obesity that are conducted on patients aged between 30 and 50, and are based in the state of Texas, United States
            ` obesity AND AREA[EligibilityMinimumAge]RANGE[30,50] AND AREA[LocationState]Texas AND AREA[LocationCountry]United States `

            IMPORTANT:
            Ensure you are using proper AREA and TILT search operations -- these are your #1 tool. 
            ALWAYS think about how you can use search to fulfill the request. Resort to manually looking at the fields only if it is IMPOSSIBLE to get a good search query.

            Examples of AREA and TILT usage:
            'Rank order the top 10 studies on heart attack in the United States' -> use TILT instead of searching the studies.
            'Find studies on VEGF as an intervention' -> use AREA[InterventionName]VEGF instead of manually looking at the field studyArmsInterventions info.
        """
        self.functions = [
            {
                "name": "study_search",
                "description": "Searches for studies on clinicaltrials.gov, and returns a list of study names and IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_term": {
                            "type": "string",
                            "description": "The search query. See the system message for operator usage for complex queries."
                        },          
                        "pageSize": {
                            "type": "integer",
                            "description": "The number of results to return. Default 10, max 20."
                        }
                    }
                }
            },
            {
                "name": "get_field_info",
                "description": "Get a specific field's information for a study given its NCT_ID and field of interest. NCT_IDs are formatted like NCT00000000.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nct_ids": {
                            "type": "string",
                            "description": """
                            The NCT_IDs of the study to get information for, separated by commas: 'NCT00000000,NCT00000001,NCT00000002'
                            """
                        },
                        "field": {
                            "type": "string",
                            "description": "The field to get information for. Available arguments: interventionAlone, studyArmsInterventions, patientConditions, studySummary, studyDesign, patientEligibility, organizations, primaryOutcomes, secondaryOutcomes, references, statusDates. Note, if InterventionAlone does not contain the intervention you are looking for, try studyArmsInterventions."
                        # need to add: official title, detailed desc (manage tokens)
                        }
                    }
                }
            },
            {
                "name": "save_csv",
                "description": "Saves a string of comma separated values to a csv file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The comma separated values to convert to a csv."
                        },
                        "title": {
                            "type": "string",
                            "description": "The title of the csv file, without the .csv extension."
                        }
                    }
                }
            }
            # Add more functions here as dictionary elements in the list.
        ]
        self.function_map = {
            "study_search": self.study_search,
            "get_field_info": self.get_field_info,
            "save_csv": self.save_csv
            # Add the actual function implementations here
        }

    def save_csv(self, text: str = None, title: str = None):
        """
        Saves a comma separated valued string directly to a csv file.
        """
        try:
            with open(title + ".csv", 'w') as f:
                f.write(text)
            return "Saved to:" + os.getcwd() + "/" + title + ".csv"
        except Exception as e:
            return "Error saving file: " + str(e)
        
    def study_search(self, query_term: str = None, pageSize: int = 10):
        """
        Searches for studies on clinicaltrials.gov, and returns a list of study names and IDs.
        This should be improved later -- not exactly the same as search on the site currently.
        """
        url = "https://www.clinicaltrials.gov/api/v2/studies"
        params = {
            "format": "json",
            "query.parser": "advanced",
            "query.term": query_term,
            "pageSize": pageSize
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            study_data = data['studies']

            # Create a list to hold study names
            study_names = []
            nct_ids = []

                        # Loop through the studies and add each name and NCT_ID to the respective list
            for study in study_data:
                study_names.append(study['protocolSection']['identificationModule']['briefTitle'])
                nct_ids.append(study['protocolSection']['identificationModule']['nctId'])

            data_out = {
                "Query Term": query_term,
                "Page Size": pageSize,
                "Number of Results": len(study_names),
                "Study Name": study_names,
                "NCT_ID": nct_ids
            }

            return data_out
        else:
            return f"Request failed with status code {response.status_code}"

    def get_field_info(self, nct_ids: str, field: str):
        # take the comma separated string of nct_ids and split it into a list
        # get rid of any whitespace
        nct_ids = nct_ids.replace(" ", "")
        nct_ids = nct_ids.split(",")
        # create a dictionary with each nct_id as keys to hold the data for each study
        data_out = {
            'nct_ids': nct_ids,  # Add nct_ids
            'field': field,  # Add field
        }
        for nct_id in nct_ids:
            data_out[nct_id] = None
        # loop through the nct_ids and get the data for each study
        for nct_id in nct_ids:
            """Get the field information for a study given its NCT_ID."""
            url = f'https://www.clinicaltrials.gov/api/v2/studies/{nct_id}'
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                
                if field == "interventionAlone":
                    try:
                        temp_output = data['derivedSection']['interventionBrowseModule']['browseLeaves']
                    except:
                        temp_output = 'No interventions found'
                elif field == "studyArmsInterventions":
                    try: 
                        temp_output = data['protocolSection']['armsInterventionsModule']['armGroups']
                    except:
                        temp_output = 'No arms or interventions found'
                elif field == "patientConditions":
                    try:
                        temp_output = data['protocolSection']['conditionsModule']['conditions']
                    except:
                        temp_output = 'No conditions found'
                elif field == "studySummary":
                    try: 
                        temp_output = data['protocolSection']['descriptionModule']['briefSummary']
                    except:
                        temp_output = 'No brief summary found'
                elif field == "studyDesign":
                    try:
                        temp_output = data['protocolSection']['designModule']
                    except:
                        temp_output = 'No study design found'
                elif field == "patientEligibility":
                    try:
                        temp_output = data['protocolSection']['eligibilityModule']
                    except:
                        temp_output = 'No eligibility criteria found'
                elif field == "organizations":
                    try:
                        temp_output = data['protocolSection']['sponsorsCollaboratorsModule']
                    except:
                        temp_output = 'No organizations found'
                elif field == "primaryOutcomes":
                    try:
                        temp_output = data['protocolSection']['outcomesModule']['primaryOutcomes']
                    except:
                        temp_output = 'No primary outcomes found'
                elif field == "secondaryOutcomes":
                    try:
                        temp_output = data['protocolSection']['outcomesModule']['secondaryOutcomes']
                    except:
                        temp_output = 'No secondary outcomes found'
                elif field == "references":
                    try:
                        temp_output = data['protocolSection']['referencesModule']['references']
                    except:
                        temp_output = 'No references found'
                elif field == "statusDates":
                    try:
                        temp_output = data['protocolSection']['statusModule']
                    except:
                        temp_output = 'No status dates found'
                else:
                    temp_output = f"Field {field} is not a valid field. Valid fields are: studyArmsInterventions, patientConditions, studySummary, studyDesign, patientEligibility, organizations, primaryOutcomes, secondaryOutcomes, references, statusDates."
            else:
                temp_output = f"Request failed with status code {response.status_code}"

            data_out[nct_id] = temp_output

        return data_out

        return self.system_message

st.title('Chatbot')
clinical_functions = ClinicalFunctions()

model_choice = st.sidebar.radio("Choose Model", ("GPT-3.5-Turbo-16K", "GPT-4"))

st.sidebar.subheader("Plugins")
choice = st.sidebar.radio("Choose Plugin", ("None", "Clinical Trials"))

if choice == "Clinical Trials":
    SYSTEM_PROMPT = (clinical_functions.system_message)
    functions = clinical_functions.functions
else:
    SYSTEM_PROMPT = "You are a helpful assistant."


if model_choice == "GPT-3.5-Turbo-16K":
    model = 'gpt-3.5-turbo-16k'
elif model_choice == "GPT-4":
    model = 'gpt-4'

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="How can I help you?"),
    ]

for msg in st.session_state.messages:
    if isinstance(msg, FunctionMessage):
        st.chat_message('💻').write(f"Result: {msg.content}")
    else:
        st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    function_call_container = st.empty()
    function_response_container = st.empty()


    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        if choice != "None":
            llm = ChatOpenAI(
                openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler], model=model, functions = functions
            )
        else:
            llm = ChatOpenAI(
                openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler], model=model
            )
        response = llm(st.session_state.messages)

        # Checking if GPT wanted to call a function
        while response.additional_kwargs and "function_call" in response.additional_kwargs:

            function_name = response.additional_kwargs['function_call']['name']
            function_to_call = clinical_functions.function_map[function_name]
            function_args = json.loads(response.additional_kwargs["function_call"]["arguments"])

            # Call the function
            with st.spinner("Calling function..."):
                function_call_message = f"Calling function {function_name} with arguments: {function_args}"
                function_call_container.chat_message("system").write(function_call_message)
                st.session_state.messages.append(ChatMessage(role="system", content=function_call_message)) 

                try:
                    function_response = function_to_call(**function_args)
                except Exception as e:
                    function_response = f"Function call failed with error: {e}"

            function_response_container.chat_message('💻').write(f"Function response: {function_response}")

            fx_message = FunctionMessage(name=function_name, content=json.dumps(function_response))
            st.session_state.messages.append(fx_message)

            response = llm(st.session_state.messages)
        
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))