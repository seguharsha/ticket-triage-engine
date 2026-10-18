"""Load test for intake. Each simulated helpdesk sends 1 ticket/s, so N users = N tickets/s.

  locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 2m \
         --csv reports/load

Run the stack with LLM_PROVIDER=fake so the test measures this system, not an LLM provider.
Afterwards: python -m scripts.processing_report --minutes 5  (worker-side latency and throughput).
"""

import json
import random
import uuid
from pathlib import Path

from locust import HttpUser, constant_throughput, task

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.jsonl"
if DATA.exists():
    TICKETS = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
else:
    TICKETS = [{"subject": f"Load test ticket {i}", "body": "The board will not load and spins forever."}
               for i in range(100)]


class Helpdesk(HttpUser):
    wait_time = constant_throughput(1)

    @task
    def submit_ticket(self):
        row = random.choice(TICKETS)
        self.client.post(
            "/tickets",
            data={"subject": row["subject"], "body": row["body"]},
            headers={"Idempotency-Key": f"load-{uuid.uuid4()}"},
            name="POST /tickets",
        )
