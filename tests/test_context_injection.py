from pydantic_ai import Agent, RunContext, capture_run_messages
from pydantic_ai.models.test import TestModel
import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart, SystemPromptPart

def test_context_injection_ephemeral_false():
    agent = Agent('test')
    
    @agent.context_injection(ephemeral=False)
    def inject_context(ctx: RunContext[None]) -> str:
        return 'injected context'
        
    result = agent.run_sync('hello')
    assert result.output == 'success (no tool calls)'

async def test_ephemeral_context_injection():
    from pydantic_ai._agent_graph import ModelRequestNode
    
    model = TestModel()
    agent = Agent(model)
    
    @agent.context_injection(ephemeral=True)
    async def inject_ephemeral(ctx: RunContext[None]) -> str:
        return 'ephemeral context'
        
    # Verify node inspection
    async with agent.iter('hello') as run:
        async for node in run:
            if isinstance(node, ModelRequestNode):
                # Verify ephemeral parts are present in the node
                assert len(node.ephemeral_request_parts) == 1
                assert node.ephemeral_request_parts[0].content == 'ephemeral context'
                
                # Verify ephemeral parts are NOT in the persistent request attached to the node (yet)
                # (although _prepare_request runs LATER, inspecting node here shows what was passed to constructor)
                user_content = node.request.parts[0].content
                assert user_content == 'hello'
                assert len(node.request.parts) == 1

    # Verify persistence - ephemeral context should NOT be in history
    with capture_run_messages() as messages:
        await agent.run('hello')
    
    assert len(messages) == 2 # 1 request, 1 response
    req = messages[0]
    # req.parts should ONLY be [UserPromptPart('hello')]
    assert len(req.parts) == 1
    assert req.parts[0].content == 'hello'
    
    # Verify what was sent to model
    class CapturingTestModel(TestModel):
        def __init__(self):
            super().__init__()
            self.captured_messages = []
            
        async def request(self, messages, model_settings, model_request_parameters):
            self.captured_messages.append(messages)
            return await super().request(messages, model_settings, model_request_parameters)
            
    capture_model = CapturingTestModel()
    agent_capture = Agent(capture_model)
    
    @agent_capture.context_injection(ephemeral=True)
    def inject_ephemeral_2(ctx: RunContext[None]) -> str:
        return 'ephemeral context 2'

    await agent_capture.run('hello')
    
    assert len(capture_model.captured_messages) == 1
    sent_msgs = capture_model.captured_messages[0]
    last_msg = sent_msgs[-1]
    # It should have ephemeral part AND user part
    assert len(last_msg.parts) == 2
    assert last_msg.parts[0].content == 'ephemeral context 2'
    assert last_msg.parts[0].part_kind == 'system-prompt'
    assert last_msg.parts[1].content == 'hello'
