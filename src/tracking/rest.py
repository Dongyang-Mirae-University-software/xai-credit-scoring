"""
MLflow 서버 REST 호출 (mlflow 패키지 없이 파이썬 기본 기능만 사용).

주소·계정은 환경 변수 MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD 에서 읽고,
없으면 레포 폴더의 .env 를 읽는다. 비밀번호는 출력하지 않는다.
"""

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENV_KEYS = ('MLFLOW_TRACKING_URI', 'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD', 'MLFLOW_EXPERIMENT_NAME')


def read_env(env_file='.env'):
    """환경 변수 우선, 없으면 .env 파일 값."""
    values = {}
    path = Path(env_file)
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    for k in ENV_KEYS:
        if os.getenv(k):
            values[k] = os.getenv(k)
    return values


class MlflowRest:
    def __init__(self, uri, username=None, password=None, opener=None):
        if not uri:
            raise ValueError("MLFLOW_TRACKING_URI 가 없습니다 (.env 확인)")
        self.uri = uri.rstrip('/')
        self._auth = None
        if username and password:
            self._auth = 'Basic ' + base64.b64encode(f'{username}:{password}'.encode()).decode()
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=60))

    @classmethod
    def from_env(cls, env_file='.env', opener=None):
        env = read_env(env_file)
        return cls(env.get('MLFLOW_TRACKING_URI'), env.get('MLFLOW_TRACKING_USERNAME'),
                   env.get('MLFLOW_TRACKING_PASSWORD'), opener=opener)

    def _request(self, method, path, body=None, data=None, content_type='application/json'):
        if body is not None:
            data = json.dumps(body).encode()
        headers = {'Content-Type': content_type}
        if self._auth:
            headers['Authorization'] = self._auth
        req = urllib.request.Request(self.uri + path, data=data, method=method, headers=headers)
        try:
            resp = self._open(req)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"MLflow 서버 응답 {e.code} ({method} {path}): 주소·계정·권한을 확인하세요") from None
        raw = resp.read()
        return json.loads(raw) if raw else {}

    def experiment_id(self, name):
        q = urllib.parse.quote(name)
        return self._request('GET', f'/api/2.0/mlflow/experiments/get-by-name?experiment_name={q}')['experiment']['experiment_id']

    def latest_run(self, experiment_id, run_name):
        """이름이 run_name 인 가장 최근 run (info + data)."""
        body = {'experiment_ids': [experiment_id], 'filter': f"attributes.run_name = '{run_name}'",
                'order_by': ['attributes.start_time DESC'], 'max_results': 1}
        runs = self._request('POST', '/api/2.0/mlflow/runs/search', body).get('runs', [])
        if not runs:
            raise RuntimeError(f"'{run_name}' run 을 찾지 못했습니다")
        return runs[0]

    def get_run(self, run_id):
        return self._request('GET', f'/api/2.0/mlflow/runs/get?run_id={urllib.parse.quote(run_id)}')['run']

    def set_tag(self, run_id, key, value):
        self._request('POST', '/api/2.0/mlflow/runs/set-tag', {'run_id': run_id, 'key': key, 'value': str(value)})

    def upload(self, artifact_uri, local_path, artifact_dir=''):
        """run 의 artifact 폴더에 파일 하나를 올린다 (서버가 mlflow-artifacts 방식일 때)."""
        prefix = 'mlflow-artifacts:/'
        if not artifact_uri.startswith(prefix):
            raise RuntimeError(f"이 서버의 artifact 주소({artifact_uri})는 REST 업로드를 지원하지 않습니다")
        base = artifact_uri[len(prefix):].lstrip('/')
        name = Path(local_path).name
        rel = '/'.join(p for p in (base, artifact_dir.strip('/'), name) if p)
        ctype = 'application/json' if name.endswith('.json') else ('image/png' if name.endswith('.png') else 'text/plain')
        self._request('PUT', '/api/2.0/mlflow-artifacts/artifacts/' + urllib.parse.quote(rel),
                      data=Path(local_path).read_bytes(), content_type=ctype)
        return rel
