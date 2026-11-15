from types import SimpleNamespace
import pytest

from glm_desktop.app import Controller


@pytest.fixture(autouse=True)
def local_directory_fixture(monkeypatch):
    # Controller unit tests do not contact production. The directory HTTP test is separate.
    monkeypatch.setattr("glm_desktop.app.public_servers", lambda _: [])


def window():
    return SimpleNamespace(api_url="https://example.com",device_index=0,distro="Ubuntu",host="",bundle="",peer="",worker_name="test",shard="0",resume=False,connected=False,stale=False)


def test_stale_connection_response_cannot_replace_current_session(monkeypatch,tmp_path):
    monkeypatch.setattr("glm_desktop.app.settings_path",lambda:tmp_path/"no-settings.json")
    ui=window()
    controller=Controller(ui)
    called=[]
    try:
        controller.generation=2
        controller.polling=True
        controller.events.put(("task",(1,lambda value:called.append(value),"old session",None,False)))
        controller.next_poll=float("inf")
        controller.tick()
        assert not called and controller.polling
    finally:
        controller.close()


def test_ready_action_is_for_this_local_instance(monkeypatch,tmp_path):
    monkeypatch.setattr("glm_desktop.app.settings_path",lambda:tmp_path/"no-settings.json")
    ui=window()
    controller=Controller(ui)
    actions=[]
    try:
        controller.instance="owned-instance"
        ui.can_ready=True
        ui.local_ready=False
        monkeypatch.setattr(controller,"action",lambda name,**values:actions.append((name,values)))
        controller.ready()
        assert actions==[("request_ready",dict(instanceId="owned-instance",ready=True))]
        ui.can_ready=False
        controller.ready()
        assert len(actions)==1
    finally:
        controller.close()


def test_debug_can_resume_exact_wizard_step_and_settings_do_not_finish_it(monkeypatch,tmp_path):
    monkeypatch.setattr("glm_desktop.app.settings_path",lambda:tmp_path/"settings.json")
    ui=window()
    controller=Controller(ui)
    try:
        ui.setup_step=2
        ui.page=3
        ui.resume_setup()
        assert ui.page==0 and ui.setup_step==2
        assert controller.save() is True
        assert ui.setup_complete is False
    finally:
        controller.close()


def test_check_spinner_lasts_for_subprocess_not_just_launch_future(monkeypatch,tmp_path):
    monkeypatch.setattr("glm_desktop.app.settings_path",lambda:tmp_path/"settings.json")
    ui=window()
    controller=Controller(ui)
    try:
        controller.jobs['check']=SimpleNamespace(running=True, stop=lambda:None)
        controller.pending=0
        controller.tick()
        assert ui.busy is True
        controller.jobs['check'].running=False
        controller.tick()
        assert ui.busy is False
    finally:
        controller.close()


def test_signin_error_fades_before_retry_is_enabled(monkeypatch,tmp_path):
    monkeypatch.setattr("glm_desktop.app.settings_path",lambda:tmp_path/"settings.json")
    now=[100.0]
    monkeypatch.setattr("glm_desktop.app.time.monotonic",lambda:now[0])
    ui=window(); controller=Controller(ui)
    try:
        controller.login_error('Incorrect password')
        controller.tick()
        assert ui.login_error_visible and ui.login_retry_blocked
        now[0]=103.8;controller.tick()
        assert not ui.login_error_visible and ui.login_retry_blocked
        now[0]=104.1;controller.tick()
        assert not ui.login_retry_blocked
    finally:
        controller.close()


def test_private_server_selection_does_not_reuse_or_publish_credentials(monkeypatch,tmp_path):
    monkeypatch.setattr("glm_desktop.app.settings_path",lambda:tmp_path/"settings.json")
    ui=window();controller=Controller(ui)
    try:
        ui.access_key='old-key';ui.admin_password='old-password'
        controller.choose_server('https://private.example','Private team')
        assert ui.server_selected and ui.selected_server_origin=='https://private.example'
        assert ui.access_key=='' and ui.admin_password=='' and not ui.connected
        assert controller.api is None
        assert not (tmp_path/'settings.json').exists()
        controller.browse_servers()
        assert not ui.server_selected and ui.page==0 and ui.setup_step==0
    finally:
        controller.close()
