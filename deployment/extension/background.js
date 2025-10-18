let enabled=false;

const config = {
    mode: "fixed_servers",
    rules: {
        singleProxy: {
            scheme: "http",
            host: "127.0.0.1",
            port: 8080
        },
        bypassList: ["<local>"]
    }
};


// Listen for messages from other parts of the extension
chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
    if (message.action === "updateProxy") {
        if (message.turnon&&!enabled){
            try {
                // Ping the local server's health check endpoint
                const response = await fetch('http://127.0.0.1:8080/health-check');
                if (!response.ok) {
                    throw new Error('Server responded with an error');
                }
            
                // If fetch is successful, the server is running.
                console.log("Proxy server is listening.");
                chrome.proxy.settings.set(
                    { value: config, scope: 'regular' },
                    () => { console.log("Proxy enabled."); }
                );
                enabled=true;
                chrome.runtime.sendMessage({action:"respond",result:"turned on"});
            
            } catch (error) {
                // If fetch fails, the server is not running.
                console.log("Failed to connect to proxy server:", error);
                chrome.runtime.sendMessage({action:"respond",result:"failed proxy"});
            }
        }
        else {
            // Turn the proxy OFF by clearing the settings
            chrome.proxy.settings.clear(
                { scope: 'regular' },
                () => { console.log("Proxy disabled."); }
            );
            enabled=false;
            chrome.runtime.sendMessage({action:"respond",result:"turned off"});
        }            
    } 
    else if (message.action==="init_mitmproxy"){
        chrome.proxy.settings.set(
            { value: config, scope: 'regular' },
            () => { console.log("Init MITM."); }
        );
        enabled=true;
    }
});

chrome.webNavigation.onBeforeNavigate.addListener(function(details){
    chrome.tabs.query({currentWindow:true, active: true},function(){
        if(details.frameId==0&& "url" in details){
            let tlu=details["url"]
            chrome.declarativeNetRequest.updateDynamicRules(
                {
                  addRules:[{
                        "id": 15,
                        "priority": 9,
                        "action": {
                            "type": "modifyHeaders",
                            "requestHeaders": [
                                {
                                "header": "afp-top-level-url",
                                "operation": "set",
                                "value": tlu
                                }
                            ]
                        },
                        "condition": {
                            "urlFilter": "*",
                            "resourceTypes": [
                                "main_frame","sub_frame","stylesheet","script","image","font","object","xmlhttprequest","ping","csp_report","media","websocket","webtransport","webbundle","other"
                            ]
                        }
                    }
                  ],
                  removeRuleIds:[15]
                }
            );
        }  
    });
});

