import unittest
import hashlib
import env

from coach import CoachClient

class TestClient(unittest.TestCase):
    def setUp(self):
        self.client = CoachClient(is_debug=True)
        self.api = env.API_KEY
        self.id = self.api[0:5]

    def login(self):
        self.client = self.client.login(self.api)        

    def test_login(self):
        self.login()

        self.assertEqual(self.client.id, self.id)
        self.assertEqual(self.client.bucket, "coach-aod")

    def test_cache_model(self):
        self.login()

        # Download the model
        self.client.cache_model("flowers", skip_match=False, model_type="frozen")
        
        # Check MD5
        frozen_graph = self.md5("flowers/frozen.pb")
        self.assertEqual(frozen_graph, "eafe442dd924aea05d1db6640927b71d")

    def test_get_model(self):
        self.login()
        # Download the model
        self.client.cache_model("flowers", skip_match=False, model_type="frozen")

        # Parse the model
        model = self.client.get_model("flowers")
        self.assertEqual(model.input_height, 224)
        self.assertEqual(model.input_width, 224)
        self.assertIsNotNone(model.graph)
        self.assertEqual(model.labels, ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])

    def test_get_model_remote(self):
        self.login()

        model = self.client.get_model_remote("flowers")
        self.assertEqual(model.input_height, 224)
        self.assertEqual(model.input_width, 224)
        self.assertIsNotNone(model.graph)
        self.assertEqual(model.labels, ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])

    def md5(self, fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

if __name__ == '__main__':
    unittest.main()