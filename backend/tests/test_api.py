import unittest
from fastapi.testclient import TestClient
from backend.main import app

class TestFastAPIRoutes(unittest.TestCase):
    """
    Test FastAPI quantitative research API endpoints.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "fastapi_quant_backend")
        self.assertEqual(data["ticker"], "^NSEI")

    def test_system_status_endpoint(self):
        resp = self.client.get("/system-status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("runtime", data)
        self.assertIn("packages", data)
        self.assertIn("database_counts", data)
        self.assertEqual(data["packages"]["fastapi"], "0.160.0")

    def test_manifest_endpoint(self):
        resp = self.client.get("/manifest")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("manifest", data)
        names = [item["name"] for item in data["manifest"]]
        self.assertIn("Python 3.10 FastAPI Backend", names)
        self.assertIn("Data Ingestion Provider", names)
        self.assertIn("SQLite Repository", names)

if __name__ == "__main__":
    unittest.main()
