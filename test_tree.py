#!/usr/bin/env python3
"""
Simple test script to verify the tree execution system works.
Run this after installing dependencies.
"""

import asyncio
import sys

async def test_basic_components():
    """Test that all components can be imported and basic operations work."""
    print("Testing Slowly v2 Tree Execution System")
    print("=" * 50)

    # Test 1: Import core models
    print("\n1. Testing imports...")
    try:
        from memory.redis_store import (
            TaskDefinition, TaskOutput, SlotDef,
            extract_slot_references, InMemoryStateStore
        )
        print("   ✅ memory.redis_store")
    except Exception as e:
        print(f"   ❌ memory.redis_store: {e}")
        return False

    try:
        from orchestration.decomposer import Decomposer, create_decomposer
        print("   ✅ orchestration.decomposer")
    except Exception as e:
        print(f"   ❌ orchestration.decomposer: {e}")
        return False

    try:
        from orchestration.assembler import Assembler, create_assembler
        print("   ✅ orchestration.assembler")
    except Exception as e:
        print(f"   ❌ orchestration.assembler: {e}")
        return False

    # Test 2: Slot reference extraction
    print("\n2. Testing slot reference extraction...")
    refs = extract_slot_references("Evaluate {antiderivative} at {point}")
    assert refs == ["antiderivative", "point"], f"Expected ['antiderivative', 'point'], got {refs}"
    print(f"   ✅ Extracted: {refs}")

    # Test 3: In-memory store
    print("\n3. Testing in-memory store...")
    store = InMemoryStateStore()
    await store.connect()

    # Create a task definition
    task = TaskDefinition(
        task_id="test_001",
        parent_task_id=None,
        slot_name="root",
        question="What is 2 + 2?",
        is_leaf=True,
        status="pending"
    )
    await store.save_definition("run_test", task)

    # Retrieve it
    loaded = await store.get_definition("run_test", "test_001")
    assert loaded is not None, "Task not found"
    assert loaded.question == "What is 2 + 2?", "Question mismatch"
    print(f"   ✅ Saved and loaded task: {loaded.task_id}")

    # Test status update
    await store.set_status("run_test", "test_001", "completed")
    status = await store.get_status("run_test", "test_001")
    assert status == "completed", f"Expected 'completed', got {status}"
    print(f"   ✅ Status updated: {status}")

    # Test 4: Task output
    print("\n4. Testing task output...")
    output = TaskOutput(
        task_id="test_001",
        output="The answer is 4",
        sources=["mental_math"],
        confidence=1.0
    )
    await store.save_output("run_test", output)

    loaded_output = await store.get_output("run_test", "test_001")
    assert loaded_output is not None, "Output not found"
    assert loaded_output.output == "The answer is 4", "Output mismatch"
    print(f"   ✅ Saved and loaded output: {loaded_output.output[:30]}...")

    # Test 5: Parent-child relationships
    print("\n5. Testing parent-child relationships...")
    parent = TaskDefinition(
        task_id="parent_001",
        parent_task_id=None,
        slot_name="root",
        question="Complex question",
        is_leaf=False,
        template=[
            SlotDef(slot="part_a", question="First part"),
            SlotDef(slot="part_b", question="Second part using {part_a}")
        ],
        status="waiting"
    )
    await store.save_definition("run_test", parent)

    child_a = TaskDefinition(
        task_id="child_a",
        parent_task_id="parent_001",
        slot_name="part_a",
        question="First part",
        is_leaf=True,
        status="completed"
    )
    await store.save_definition("run_test", child_a)
    await store.save_output("run_test", TaskOutput(
        task_id="child_a",
        output="Result A",
        sources=[],
        confidence=0.9
    ))

    child_b = TaskDefinition(
        task_id="child_b",
        parent_task_id="parent_001",
        slot_name="part_b",
        question="Second part using {part_a}",
        is_leaf=True,
        status="pending"
    )
    await store.save_definition("run_test", child_b)

    children = await store.get_children_ids("run_test", "parent_001")
    assert len(children) == 2, f"Expected 2 children, got {len(children)}"
    print(f"   ✅ Children found: {children}")

    # Check all_children_completed
    all_done = await store.all_children_completed("run_test", "parent_001")
    assert all_done == False, "Should not be complete yet"
    print(f"   ✅ all_children_completed (partial): {all_done}")

    # Complete child_b
    await store.set_status("run_test", "child_b", "completed")
    await store.save_output("run_test", TaskOutput(
        task_id="child_b",
        output="Result B",
        sources=[],
        confidence=0.8
    ))

    all_done = await store.all_children_completed("run_test", "parent_001")
    assert all_done == True, "Should be complete now"
    print(f"   ✅ all_children_completed (full): {all_done}")

    # Get children outputs
    outputs = await store.get_children_outputs("run_test", "parent_001")
    assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
    assert "part_a" in outputs and "part_b" in outputs
    print(f"   ✅ Children outputs by slot: {list(outputs.keys())}")

    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)

    print("\n📋 Next steps:")
    print("   1. Install all dependencies:")
    print("      pip install -r requirements.txt")
    print("")
    print("   2. Start the web server:")
    print("      python server.py")
    print("")
    print("   3. Or run CLI:")
    print("      python main_v2.py \"Your question here\"")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_basic_components())
    sys.exit(0 if success else 1)
