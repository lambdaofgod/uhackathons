from tinydb import TinyDB, Query
from app.models import Question
from fastapi.encoders import jsonable_encoder


class QuestionThread:

    def get_or_create_thread_id(self, question: str) -> str:
        if self.thread_id:
            return self.thread_id
        else:
            return question

    def __init__(self):
        self.thread_id = None
        self.answers = []
        self._storage = ThreadStorage()

    def add_answer(self, answer: Question) -> None:
        if not self.thread_id:
            self.thread_id = answer.thread_id
        self.answers.append(answer)

    def reset_thread(self):
        for old_answer in self.answers:
            self._storage.add_answer(old_answer)
        self.answers = []
        self.thread_id = None

    def get_thread_answers(self, thread_id: str):
        return self._storage.get_thread_answers(thread_id)


class ThreadStorage:

    def __init__(self, db_path: str = 'thread_db.json') -> None:
        self._db_path = db_path
        self._db = TinyDB(db_path)
        self._answers_table = self._db.table('answers')

    def add_answer(self, answer: Question):
        self._answers_table.insert(jsonable_encoder(answer))

    def get_thread_answers(self, thread_id: str):
        data = self._answers_table.search(Query().thread_id == thread_id)
        answers = [Question(**answer) for answer in data]
        return sorted(answers, key=lambda x: x.created_at)
