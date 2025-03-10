import javafx.application.Application;

public class Main extends Application {

    @Override
    public void start(javafx.stage.Stage primaryStage) {
        App app = new App();
        app.initialize(primaryStage);
    }

    public static void main(String[] args) {
        launch(args);  
    }
}
