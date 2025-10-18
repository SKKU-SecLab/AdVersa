package com.example;

// CORRECTED IMPORTS using the official hex.genmodel package
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

import org.json.JSONObject;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import hex.genmodel.MojoModel;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.prediction.BinomialModelPrediction;


public class App {
    private static EasyPredictModelWrapper model;
    
    // IMPORTANT: Change this string to match the name of YOUR .zip file!
    private static final String MODEL_FILE_PATH = "AdVersaModel.zip"; 

    public static void main(String[] args) throws Exception {
        // This logic does not change at all
        java.net.URL modelUrl = App.class.getClassLoader().getResource("AdVersaModel.zip");
        System.out.println("Loading MOJO model from: " + MODEL_FILE_PATH);
        model = new EasyPredictModelWrapper(MojoModel.load(modelUrl.getPath()));
        System.out.println("Model loaded successfully.");

        HttpServer server = HttpServer.create(new InetSocketAddress(9090), 0);
        server.createContext("/predict", new PredictionHandler());
        server.setExecutor(null); 

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nShutdown signal received, shutting down server...");
            server.stop(1); // Stop the server, allowing 1 second for existing connections to close
            System.out.println("Server stopped.");
            // You could add other cleanup code here if needed
        }));
        
        server.start();

        System.out.println("🚀 Server started and listening on http://127.0.0.1:9090/predict");
    }

    static class PredictionHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            // This logic does not change at all
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "{\"error\":\"Method Not Allowed\"}");
                return;
            }

            try {
                InputStream is = exchange.getRequestBody();
                String requestBody = new String(is.readAllBytes(), StandardCharsets.UTF_8);

                JSONObject jsonRequest = new JSONObject(requestBody);
                RowData row = new RowData();
                for (String key : jsonRequest.keySet()) {
                    row.put(key, jsonRequest.get(key).toString());
                }

                BinomialModelPrediction prediction = model.predictBinomial(row);

                JSONObject jsonResponse = new JSONObject();
                jsonResponse.put("predictedLabel", prediction.label);
                jsonResponse.put("classProbabilities", prediction.classProbabilities);

                sendResponse(exchange, 200, jsonResponse.toString());

            } catch (Exception e) {
                sendResponse(exchange, 400, "{\"error\":\"Bad Request\"}");
            }
        }

        private void sendResponse(HttpExchange exchange, int statusCode, String response) throws IOException {
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(statusCode, response.length());
            OutputStream os = exchange.getResponseBody();
            os.write(response.getBytes());
            os.close();
        }
    }
}