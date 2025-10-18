document.addEventListener("DOMContentLoaded", () => {
  
    const toggle_button=document.getElementById("toggle_button");
    const toggle_element=document.getElementById("toggle_element");
    const toggle_description=document.getElementById("toggle_description");
    const init_btn=document.getElementById("init_btn");

    init_btn.addEventListener("click", function(){
        chrome.runtime.sendMessage({ action: "init_mitmproxy" });
    });

    chrome.storage.sync.get({"toggle":true},function(res){
        if(res.toggle){
            chrome.runtime.sendMessage({ action: "updateProxy", turnon: true });
            toggle_element.checked=true;
            toggle_description.innerText="Proxy ON";
        }
        else{
            toggle_element.checked=false;
            toggle_description.innerText="Proxy OFF";
        }
    });

    chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
        if(message.action==="respond"){
            console.log(message.result);
            if(message.result==="failed proxy"){
                toggle_element.checked=false;
                toggle_description.innerText="Proxy OFF";
                chrome.storage.sync.set({"toggle":false});
            }
        }
    });
    
    toggle_button.addEventListener("click", function(){
        console.log(toggle_element.checked);
        if(toggle_element.checked){
            console.log("turn off");
            chrome.runtime.sendMessage({ action: "updateProxy", turnon: false });
            chrome.storage.sync.set({"toggle":false});
            toggle_description.innerText="Proxy OFF";
        }
        else{
            console.log("turn on");
            chrome.runtime.sendMessage({ action: "updateProxy", turnon: true });
            chrome.storage.sync.set({"toggle":true});
            toggle_description.innerText="Proxy ON";
        }
    });
});
                