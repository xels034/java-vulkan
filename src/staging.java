import java.util.Arrays;

import org.lwjgl.system.Configuration;

public class staging {
  public static void main(String[] args) {
    var lwjglDebug = false;

    Configuration.MEMORY_ALLOCATOR.set("rpmalloc");

    Configuration.DEBUG.set(lwjglDebug);
    Configuration.DEBUG_MEMORY_ALLOCATOR.set(lwjglDebug);
    Configuration.DEBUG_STACK.set(lwjglDebug);


    System.out.println("Args: " + Arrays.toString(args));
    var debug = (Boolean.parseBoolean(System.getProperty("debug", "false")) || Arrays.stream(args).filter(s -> s.equals("-debug")).count() > 0);



    var app = new Application(1920, 1080, debug);

    app.init();
    app.run();
    app.dispose();
  }
}