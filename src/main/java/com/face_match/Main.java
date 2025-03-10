package com.face_match;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.stage.Stage;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) {
        // Create a Label with "Hello, World!"
        Label helloWorldLabel = new Label("Hello, World!");

        // Create a Scene, set the label as the root node, and set the size
        Scene scene = new Scene(helloWorldLabel, 300, 200);

        // Set the stage (window) properties
        primaryStage.setTitle("JavaFX Hello World");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);  // Launch the JavaFX application
    }
}
