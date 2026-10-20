import json

from RAG.rl.trajectory_buffer import TrajectoryBuffer, TrajectoryRecord


def _record(question_id: int, *, answer: str = "answer") -> TrajectoryRecord:
    return TrajectoryRecord(
        question_id=question_id,
        split="train",
        route_name="all_t1",
        route_key="0-0-0-0",
        actions=[0, 0, 0, 0],
        f1_score=1.0,
        reward=1.0,
        answer=answer,
        ground_truth="answer",
        operator_version=1,
        workflow_identity="workflow-v1",
    )


class _CountingTrajectoryBuffer(TrajectoryBuffer):
    def __init__(self, *args, **kwargs):
        self.key_calls = 0
        super().__init__(*args, **kwargs)

    def _key(self, record):
        self.key_calls += 1
        return super()._key(record)


def _write_records(path, count: int = 200) -> None:
    path.write_text(
        json.dumps([_record(question_id).to_dict() for question_id in range(count)]),
        encoding="utf-8",
    )


def test_loaded_upsert_does_not_scan_the_existing_buffer(tmp_path):
    path = tmp_path / "new1.json"
    _write_records(path)
    buffer = _CountingTrajectoryBuffer(
        path,
        workflow_identity="workflow-v1",
        operator_version=1,
    )

    buffer.key_calls = 0
    replaced = buffer.upsert_by_question_route(_record(199, answer="updated"))

    assert replaced is True
    assert buffer.key_calls <= 3
    assert len(buffer.records) == 200
    assert buffer.records[-1].answer == "updated"


def test_remove_question_ids_keeps_constant_time_upsert_lookup(tmp_path):
    path = tmp_path / "new1.json"
    _write_records(path)
    buffer = _CountingTrajectoryBuffer(
        path,
        workflow_identity="workflow-v1",
        operator_version=1,
    )

    assert buffer.remove_question_ids(set(range(50))) == 50
    buffer.key_calls = 0
    replaced = buffer.upsert_by_question_route(_record(199, answer="updated"))

    assert replaced is True
    assert buffer.key_calls <= 3
    assert len(buffer.records) == 150
    assert buffer.records[-1].answer == "updated"


def test_save_does_not_rewrite_an_unchanged_loaded_buffer(tmp_path):
    path = tmp_path / "new1.json"
    original = json.dumps([_record(1).to_dict()], separators=(",", ":"))
    path.write_text(original, encoding="utf-8")
    buffer = TrajectoryBuffer(
        path,
        workflow_identity="workflow-v1",
        operator_version=1,
    )

    assert buffer.upsert_by_question_route(_record(1)) is True
    buffer.save()

    assert path.read_text(encoding="utf-8") == original

    buffer.upsert_by_question_route(_record(1, answer="updated"))
    buffer.save()

    assert json.loads(path.read_text(encoding="utf-8"))[0]["answer"] == "updated"
