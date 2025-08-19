# Requirements: Sequential Tool Calling for AI Generator

## Feature Overview
Refactor `backend/ai_generator.py` to support sequential tool calling where Claude can make up to 2 tool calls in separate API rounds, allowing for complex multi-step reasoning and comparative analysis.

## User Stories and Acceptance Criteria

### Story 1: Multi-Step Search Capability
**As a** user asking complex questions requiring multiple searches  
**I want** Claude to be able to search different courses or lessons sequentially  
**So that** I can get comprehensive answers that compare information across multiple sources

**Acceptance Criteria:**
- **WHEN** a user asks a question requiring multiple searches, **THEN** Claude **SHALL** be able to make up to 2 sequential tool calls
- **WHEN** Claude receives results from the first tool call, **THEN** it **SHALL** be able to reason about those results and decide if a second search is needed
- **WHEN** Claude makes a second tool call, **THEN** it **SHALL** have access to the conversation context including the first tool call results

### Story 2: Conversation Context Preservation
**As a** system processing sequential tool calls  
**I want** conversation context to be preserved between tool rounds  
**So that** Claude can make informed decisions about subsequent searches

**Acceptance Criteria:**
- **WHEN** Claude makes the first tool call, **THEN** the conversation history **SHALL** be preserved for the second round
- **WHEN** Claude makes the second tool call, **THEN** it **SHALL** have access to all previous messages in the conversation thread
- **WHEN** tool execution occurs, **THEN** all tool results **SHALL** be properly formatted and included in the conversation context

### Story 3: Termination Conditions
**As a** system preventing infinite loops  
**I want** clear termination conditions for tool calling sequences  
**So that** the system remains responsive and doesn't waste API calls

**Acceptance Criteria:**
- **WHEN** 2 tool calling rounds have been completed, **THEN** the system **SHALL** terminate and return the final response
- **WHEN** Claude's response contains no tool_use blocks, **THEN** the system **SHALL** terminate and return that response
- **WHEN** a tool call fails with an error, **THEN** the system **SHALL** terminate and return an error response

### Story 4: Error Handling and Recovery
**As a** system handling tool execution errors  
**I want** graceful error handling at each step  
**So that** users receive meaningful feedback when tool calls fail

**Acceptance Criteria:**
- **WHEN** a tool execution fails, **THEN** the system **SHALL** provide a descriptive error message
- **WHEN** the first tool call fails, **THEN** the system **SHALL** terminate without attempting a second call
- **WHEN** the second tool call fails, **THEN** the system **SHALL** return results from the first successful call plus an error message

### Story 5: System Prompt Updates
**As a** system supporting sequential tool calling  
**I want** updated system prompts that guide Claude's multi-step behavior  
**So that** Claude uses the new capability effectively

**Acceptance Criteria:**
- **WHEN** the system prompt is updated, **THEN** it **SHALL** remove the "One tool use per query maximum" restriction
- **WHEN** the system prompt is updated, **THEN** it **SHALL** include guidance on when to make multiple tool calls
- **WHEN** the system prompt is updated, **THEN** it **SHALL** instruct Claude to reason about whether additional searches are needed

### Story 6: API Call Pattern Optimization
**As a** system making multiple API calls  
**I want** efficient API call patterns that minimize latency  
**So that** users experience reasonable response times

**Acceptance Criteria:**
- **WHEN** sequential tool calls are made, **THEN** each API call **SHALL** be a separate request to Claude's API
- **WHEN** building API parameters, **THEN** the system **SHALL** reuse base parameters efficiently
- **WHEN** managing conversation state, **THEN** the system **SHALL** append messages incrementally without rebuilding the entire conversation

## Use Cases Enabled

1. **Comparative Analysis**: "Compare the authentication approaches in the MCP course vs the Web Development course"
2. **Multi-Course Integration**: "How do the concepts from Lesson 1 of Course A relate to Lesson 3 of Course B?"
3. **Progressive Information Gathering**: "Tell me about user authentication, then explain how it applies to the specific framework mentioned in Course X"
4. **Cross-Reference Validation**: "Find information about topic X, then search for any contradictory information in other courses"

## Technical Constraints

- Maximum 2 sequential tool calling rounds per user query
- Preserve existing conversation history limits (2 previous exchanges)
- Maintain compatibility with existing tool definitions and ToolManager
- Preserve source tracking functionality for UI display
- No changes to the core tool execution logic in search_tools.py