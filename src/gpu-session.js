// Web Locks serialize this origin's tabs; native processes and other origins
// still need coordination by the operator.
export async function acquireGpuSession(locks=globalThis.navigator?.locks) {
  if(!locks)throw new Error('Web Locks are required to run this GPU experiment.');
  let release,accept,reject;
  const held=new Promise(resolve=>{release=resolve;});
  const acquired=new Promise((resolve,onError)=>{accept=resolve;reject=onError;});
  locks.request('gs-game-engine-gpu',{mode:'exclusive',ifAvailable:true},async lock=>{
    if(!lock){reject(new Error('Another tab on this origin owns the GPU session. Close that experiment tab first.'));return;}
    accept(release);
    await held;
  }).catch(reject);
  return acquired;
}
