"""Coordinate idle-only restarts without interrupting Studio work."""
import threading
import uuid

from .store import Conflict


class Lifecycle:
    def __init__(self, service, enabled=False):
        self.service = service
        self.enabled = enabled and not service.owner_url
        self.boot_id = uuid.uuid4().hex
        self.restarting = False
        self.writes = 0
        self.lock = threading.Lock()

    def status(self):
        return {'boot_id': self.boot_id, 'restart_available': self.enabled,
                'restarting': self.restarting}

    def enter_write(self):
        with self.lock:
            if self.restarting:
                raise Conflict('LoRA Studio is restarting. Wait for it to reconnect, then try again.')
            self.writes += 1

    def leave_write(self):
        with self.lock:
            self.writes -= 1

    def request_restart(self):
        service = self.service
        with self.lock:
            if not self.enabled:
                raise Conflict('Restart is unavailable in this server. Start LoRA Studio with start.bat or python server.py.')
            if self.writes > 1:
                raise Conflict('Wait for other changes to finish saving, then restart.')
            if not service.job_lock.acquire(blocking=False):
                raise Conflict('Wait for active Studio work to finish before restarting.')
            try:
                jobs = service.store.items('jobs')
                if any(j['status'] in {'queued', 'running', 'stopping'} or
                       (j['kind'] == 'train' and j['status'] == 'interrupted') for j in jobs):
                    raise Conflict('Finish active jobs and resolve interrupted training runs before restarting.')
                setup = service.setup
                status = setup.status()
                if (setup.check_lock.locked() or
                    any(status.get(key, {}).get('status') in {'queued', 'running', 'stopping'}
                        for key in ('install', 'promptstudio_install')) or
                    (setup.process is not None and setup.process.poll() is None)):
                    raise Conflict('Wait for setup or installation to finish before restarting.')
                if service.comparisons.active:
                    raise Conflict('Wait for checkpoint comparisons to finish before restarting.')
                self.restarting = True
            except BaseException:
                service.job_lock.release()
                raise
            # Retain the job reservation until this server exits.
            return self.status()
