import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent))
import ssq_data_sources as sources
import ssq_http as http


class HttpTransportTests(unittest.TestCase):
    def test_data_sources_preserves_http_compatibility_exports(self):
        self.assertIs(sources.create_http_session, http.create_http_session)
        self.assertIs(sources.fetch_text, http.fetch_text)
        self.assertIs(sources.REQUEST_HEADERS, http.REQUEST_HEADERS)
        self.assertEqual(
            sources.REQUEST_TIMEOUT_SECONDS,
            http.REQUEST_TIMEOUT_SECONDS,
        )

    def test_session_retries_transient_get_failures(self):
        session = http.create_http_session()
        retry = session.get_adapter('https://').max_retries

        self.assertEqual(retry.total, 3)
        self.assertEqual(tuple(retry.allowed_methods), ('GET',))
        self.assertIn(429, retry.status_forcelist)

    def test_fetch_text_applies_request_policy_and_encoding(self):
        response = SimpleNamespace(
            text='draw data',
            encoding=None,
            raise_for_status=Mock(),
        )
        session = SimpleNamespace(get=Mock(return_value=response))

        content = http.fetch_text(
            'https://example.test/draws.txt',
            session=session,
            encoding='utf-8',
        )

        self.assertEqual(content, 'draw data')
        self.assertEqual(response.encoding, 'utf-8')
        response.raise_for_status.assert_called_once_with()
        session.get.assert_called_once_with(
            'https://example.test/draws.txt',
            headers=http.REQUEST_HEADERS,
            timeout=http.REQUEST_TIMEOUT_SECONDS,
        )


if __name__ == '__main__':
    unittest.main()
