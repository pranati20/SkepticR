import os
import tempfile

# keep every test (including the Librarian ones) away from the real paper library and the network
os.environ.setdefault("SKEPTIC_LIBRARY_PATH", os.path.join(tempfile.mkdtemp(), "papers.json"))
