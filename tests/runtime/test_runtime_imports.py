import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from textwrap import dedent


def test_runtime_snapshot_round_trip_does_not_load_engine_or_register_nodes() -> None:
    script = dedent(
        """
        import json
        import sys

        from graphon.runtime import RuntimeState, VariablePool

        state = RuntimeState(
            variable_pool=VariablePool(), start_at=0, workflow_id="workflow",
            outputs={"answer": "saved"},
        )
        state.ready_queue.loads(json.dumps({
            "type": "InMemoryReadyQueue", "version": "1.0", "items": ["queued"],
        }))
        restored = RuntimeState.from_snapshot(state.dumps())
        assert restored.outputs == {"answer": "saved"}
        assert restored.ready_queue.dumps() == state.ready_queue.dumps()
        assert json.loads(restored.dumps())["version"] == "3.0"
        loaded_engine_modules = [
            name for name in sys.modules
            if name == "graphon.engine" or name.startswith("graphon.engine.")
        ]
        assert not loaded_engine_modules, loaded_engine_modules

        from graphon.nodes.base.node import Node

        assert not Node.get_node_type_classes_mapping()

        from graphon.runtime import ready_queue
        from graphon.engine import ready_queue as legacy_queue
        from graphon.engine.ready_queue import entities, in_memory

        assert legacy_queue.ReadyQueue is ready_queue.ReadyQueue
        assert legacy_queue.ReadyTask is entities.ReadyTask is ready_queue.ReadyTask
        assert legacy_queue.StartTask is entities.StartTask is ready_queue.StartTask
        assert legacy_queue.ResumeTask is entities.ResumeTask is ready_queue.ResumeTask
        assert legacy_queue.InMemoryReadyQueue is in_memory.InMemoryReadyQueue
        assert legacy_queue.InMemoryReadyQueue is ready_queue.InMemoryReadyQueue
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]


def test_dsl_runs_without_optional_document_and_tokenizer_packages() -> None:
    """Core installs omit the ``documents`` and ``transformers`` extras.

    Single-threaded runtimes such as Pyodide cannot install those packages, so
    importing the DSL and running a graph inline must not load them.
    """
    script = dedent(
        """
        import importlib.abc
        import sys

        OPTIONAL = {
            "defusedxml", "docx", "odfdo", "pandas", "pypdf", "pypdfium2",
            "transformers", "unstructured", "webvtt",
        }

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] in OPTIONAL:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockOptional())

        from graphon.dsl import loads
        from graphon.engine_events.graph import GraphRunSucceededEvent

        dsl = '''
        kind: graph
        graph:
          nodes:
            - id: start
              data: {type: start, variables: []}
            - id: answer
              data: {type: answer, answer: "Hello {{#start.name#}}"}
          edges:
            - {source: start, target: answer}
        '''
        engine = loads(dsl, start_inputs={"name": "Graphon"}, workers=0)
        terminal = list(engine.run())[-1]
        assert isinstance(terminal, GraphRunSucceededEvent), terminal
        assert terminal.outputs["answer"] == "Hello Graphon"
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]
