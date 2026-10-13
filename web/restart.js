// Reconnect in place so open forms and browser drafts remain available.
export function bindRestart({save, onConnected, toast}) {
  const button = document.querySelector('#restart-studio');
  const message = document.querySelector('#restart-status');
  async function request(path, method='GET') {
    const response = await fetch('/api'+path, {
      method, signal: AbortSignal.timeout(5000),
      ...(method==='POST'?{headers:{'Content-Type':'application/json'},body:'{}'}:{}),
    });
    const data = await response.json();
    if(!response.ok) {
      const error = new Error(response.status===404
        ? 'Start LoRA Studio once from its launcher to enable Restart.'
        : data.error||'Restart failed.');
      error.status=response.status;
      throw error;
    }
    return data;
  }
  button.addEventListener('click', async()=>{
    button.disabled=true;
    message.hidden=false;
    message.textContent='Saving changes…';
    try {
      await save();
      const before=await request('/runtime');
      if(!before.restart_available)throw new Error('Restart is unavailable. Start LoRA Studio from its launcher.');
      message.textContent='Restarting LoRA Studio… Waiting to reconnect.';
      try {await request('/restart','POST');}
      catch(error) {if(error.status)throw error; /* The server may close before replying. */}
      const deadline=Date.now()+60000;
      while(Date.now()<deadline) {
        await new Promise(resolve=>setTimeout(resolve,700));
        let current;
        try {current=await request('/runtime');} catch {continue;}
        if(current.boot_id!==before.boot_id&&!current.restarting) {
          message.textContent='LoRA Studio restarted. Your open forms are preserved.';
          toast('LoRA Studio restarted.');
          onConnected();
          return;
        }
      }
      throw new Error('Could not reconnect. Check the Studio console, then refresh status or reopen the launcher.');
    } catch(error) {
      message.textContent=error.message;
      toast(error.message,true);
    } finally {button.disabled=false;}
  });
}
