import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.glfw.GLFWVulkan.*;
import static org.lwjgl.vulkan.VK13.*;
import static org.lwjgl.vulkan.KHRSurface.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.lwjgl.vulkan.EXTDebugUtils.*;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.CustomBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.NativeResource;
import org.lwjgl.system.NativeType;
import org.lwjgl.util.shaderc.Shaderc;
import org.lwjgl.vulkan.*;

public class Application {

  //Use object wrapper as a way to encode optionals. An Optional<Integer> is just a joke of a pointer to a pointer to an int
  private static record QueueFamilyConfig (Integer                  graphics,     Integer              present,   Integer           compute,      Integer         transfer                    ){}
  private static record SwapChainConfig   (VkSurfaceCapabilitiesKHR capabilities, VkSurfaceFormatKHR[] formats,   int[]             presentModes                                              ){}
  private static record DeviceInfo        (int                      score,        VkPhysicalDevice     device,    QueueFamilyConfig queues,       SwapChainConfig swapchain, String deviceName){}
  private static record SwapChainExtra    (VkSurfaceFormatKHR       usedFormat,   VkExtent2D           usedExtent                                                                             ){}

  private long window;
  private int  width, height;
  private boolean debug;

  private long shaderCompiler;
  private long compilerOptions;

  private long vertModule;
  private long fragModule;

  private VkDebugUtilsMessengerCallbackEXT dbgCallback;
  private VkInstance instance;

  private long debugMessenger;
  private long surface;
  private long swapChain;
  private long pipelineLayout;
  private long renderPass;
  private long pipeline;
  private long commandPool;

  private long imgAvailableSem;
  private long renderFinishedSem;
  private long inFlightFence;

  private VkCommandBuffer cmdBuffer;

  private long[] swapChainImages;
  private long[] swapChainImageViews;
  private long[] swapChainFramebuffers;

  private DeviceInfo     deviceInfo;
  private SwapChainExtra deviceInfoExtra;
  private VkDevice       device;

  //commands must be put into queues. Certain queue families only accept certain commands.
  //They are owned by the device, so no need for an explicit free
  private VkQueue queueGraphics; //for drawing commands (also accepts transfer commands)
  private VkQueue queuePresent;  //for interacting with the swap chain

  private String[] requredExtensions;
  private String[] requiredValidationLayers;
  private Set<String> requiredDeviceExtensions;

  private List<CustomBuffer<?>> ownedMemoryCustom;
  private List<Buffer> ownedMemoryNIO;
  private List<NativeResource> ownedMemoryObj;

  public Application(int w, int h, boolean d){
    width  = w;
    height = h;
    debug  = d;

    ownedMemoryCustom = new ArrayList<>();
    ownedMemoryNIO    = new ArrayList<>();
    ownedMemoryObj    = new ArrayList<>();

    if(debug) System.out.println("Using debug");

    if(debug) {
      requredExtensions        = new String[]{VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
      requiredValidationLayers = new String[]{"VK_LAYER_KHRONOS_validation"};
    }
    else {
      requredExtensions        = new String[]{};
      requiredValidationLayers = new String[]{};
    }

    requiredDeviceExtensions = new HashSet<>();
    requiredDeviceExtensions.add(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  /**
   * Tracks heap allocated direct bytebuffers that must be freed at the end of the applications lifetime
   * @param <T> the type of buffer to track
   * @param mem the buffer to track
   * @return the tracked buffer for chaining
   */
  private <T extends PointerBuffer> T trackMem(T mem){
    ownedMemoryCustom.add(mem);
    return mem;
  }

  @SuppressWarnings("unused")
  private <T extends Buffer> T trackMem(T mem){
    ownedMemoryNIO.add(mem);
    return mem;
  }

  private <T extends NativeResource> T trackMem(T mem){
    ownedMemoryObj.add(mem);
    return mem;
  }

  public void init(){

    initShaderC();
    initWindow(); //typical glfw window creation
    initVulkan();
  }

  private void initShaderC(){
    shaderCompiler  = Shaderc.shaderc_compiler_initialize();
    compilerOptions = Shaderc.shaderc_compile_options_initialize();

    Shaderc.shaderc_compile_options_set_auto_map_locations(compilerOptions, true);
    Shaderc.shaderc_compile_options_set_auto_bind_uniforms(compilerOptions, true);
    Shaderc.shaderc_compile_options_set_nan_clamp         (compilerOptions, true);
    Shaderc.shaderc_compile_options_set_source_language   (compilerOptions, Shaderc.shaderc_source_language_glsl);
    Shaderc.shaderc_compile_options_set_optimization_level(compilerOptions, Shaderc.shaderc_optimization_level_zero);//keep uniform names for opengl linking to work
    Shaderc.shaderc_compile_options_set_target_env        (compilerOptions, Shaderc.shaderc_target_env_vulkan, Shaderc.shaderc_env_version_vulkan_1_3); //some built-ins are different from glsl

  }

  private void initWindow(){
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); //todo tutorials says this will be handled later

    window = glfwCreateWindow(width, height, "java-vulkan", MemoryUtil.NULL, MemoryUtil.NULL);

    if(!glfwVulkanSupported()) throw new RuntimeException("glfw doesn't support vulkan");
  }

  private void initVulkan(){
    createInstance();
    createSurface();
    selectPhyscialDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createPipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();
  }

  private PointerBuffer getInstanceExtensionsPtr(){
    var glfwPtr = glfwGetRequiredInstanceExtensions();

    PointerBuffer ptr;
    if(requredExtensions.length == 0){
      ptr = glfwPtr;

    }else{
      var extCnt = glfwPtr.remaining() + requredExtensions.length;
      var stack = MemoryStack.stackGet();

      ptr = stack.mallocPointer(extCnt);
      ptr.put(glfwPtr);
      for(var ext : requredExtensions){
        ptr.put(stack.UTF8(ext));
      }
    }

    System.out.println("exts needed: " + ptr.capacity());
    return ptr.rewind();
  }

  private PointerBuffer getDeviceExtensionsPtr(){
    //old versions required to set validation layer extensions separatley for logical devices too. this is no longer necessary

    var stack = MemoryStack.stackGet();
    var ptr = stack.mallocPointer(requiredDeviceExtensions.size());

    for(var ext : requiredDeviceExtensions) ptr.put(MemoryStack.stackUTF8(ext));

    return ptr.rewind();
  }

  private void checkValidationLayers(){
    var stack = MemoryStack.stackGet();

    var layerCntPtr = new int[1];
    vkEnumerateInstanceLayerProperties(layerCntPtr, null);
    var layerCnt = layerCntPtr[0];
    System.out.println("Found " + layerCnt + " layer properties");

    var layerPropPtr = VkLayerProperties.calloc(layerCnt, stack);
    var result = vkEnumerateInstanceLayerProperties(layerCntPtr, layerPropPtr);

    switch(result){
      case VK_INCOMPLETE : System.out.println("Not all extensions were filled in"); break;
      case VK_SUCCESS    : System.out.println("Extension list retrieved properly"); break;
    }

    for(var req_layer : requiredValidationLayers){
      var found = false;
      while(layerPropPtr.remaining() > 0){
        var lyr = layerPropPtr.get();
        if(lyr.layerNameString().equals(req_layer)) {
          found = true;
          System.out.println("Layer " + req_layer + ": Ok!");
          break;
        }
      }

      if(!found) throw new RuntimeException("Required layer " + req_layer + " missing");
    }
  }

  private PointerBuffer getValidationLayersPtr(){
    if(requiredValidationLayers.length == 0) return null;
    checkValidationLayers();

    var stack = MemoryStack.stackGet();

    var ptr = stack.mallocPointer(requiredValidationLayers.length);

    for(var lyr : requiredValidationLayers){
      ptr.put(stack.UTF8(lyr));
      System.out.println("Request layer " + lyr);
    }

    return ptr.rewind();
  }

  private VkDebugUtilsMessengerCreateInfoEXT getDebugCreateInfo(){
    if(!debug) return null;

    //while the API allows passing in a lambda directly into the struct, it isn't possible to free the native callback object afterwards.
    //While this leak is insignificant, the Debug Memory ALlocator complains, so create the proper object that can be tracked & freed
    dbgCallback = VkDebugUtilsMessengerCallbackEXT.create((VkDebugUtilsMessengerCallbackEXTI)this::debugCallback);

    return VkDebugUtilsMessengerCreateInfoEXT.calloc(MemoryStack.stackGet())
    .sType(VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT)
    .messageSeverity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    .messageType(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
    .pfnUserCallback(dbgCallback)
    .pUserData(MemoryUtil.NULL);
  }

  private @NativeType("VkBool32") int debugCallback(@NativeType("VkDebugUtilsMessageSeverityFlagBitsEXT") int messageSeverity, @NativeType("VkDebugUtilsMessageTypeFlagsEXT") int messageTypes, @NativeType("VkDebugUtilsMessengerCallbackDataEXT const *") long pCallbackData, @NativeType("void *") long pUserData){
    var callbackData = VkDebugUtilsMessengerCallbackDataEXT.create(pCallbackData);
    System.err.println(callbackData.pMessageString());
    return VK_FALSE;
  }

  private void createInstance(){
    /*

    The instance is the API entry point, describing what API level is targeted, as well as some information about the application that will use the API
    This offers the ability to query physcial devices which are available, as well as their proptierts. From that a logical device can be created that configures the selected physical device for use with the API

    The instance has several properties:
    1) Application meta-data that might help the driver to find optimizations. Such as the application name (name of a game executable), used engine (e.g. Unreal Engine), etc.
    2) Requered extensions. Vulkan is a platform agnostiv API, and "naked" vulkan has no concepts of a display output for example. So the logical device will almost always need some required extensions,
       which have to be requested explicitly. Some extensions might be debugging specific.
    3) Required layers. Layers are API layers on top of the calls that get to the driver. Layers canredirect, record, alter, etc. the API call to achieve some needs. The typical layer would be
       the validation layer, which offers debugging / profiling setup for for the API. This mimics what openGL does by default, as the default vulkan API doesn't do validation, to focus on performance


    Memory rules using java with the vulkan API:
      Many methods require the passing of different kind of buffers, and setting them up as heap-buffers (direct buffers, which are also on the heap, just not visible to the GC) can be tedious / slow.
      As a utility there is "stack allocation" which is done via a thread-local byte array that acts as a "stack", and the stack-frame is usually a try-with-resources block. (THough explicit push/pop
      is also supported)

      That means a stack-frame might be current accross different method calls. For that there is the MemoryStack.stackGet() method, which returns the current thread local stack frame entered since the lat
      MemoryStack.stackPush()

      This memory stack is used for different vulkan API calls. A bit oddly, there are explicit overloads for using a (non-configurable?) heap allocator, or a specified stack allocator.
    */

    try (var stack = MemoryStack.stackPush()){
      var extensionsPtr = getInstanceExtensionsPtr();
      var layersPtr     = getValidationLayersPtr();

      var debugInfo     = getDebugCreateInfo(); //the debug createInfo can be used in 2 places. For an explicit debugMessenger for callbacks (which requires an already created instance)
                                                //and as a next-pointer during instance creation, to cover that specifically as well

      var appInfo = VkApplicationInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
        .pApplicationName(MemoryStack.stackUTF8("java-vulkan"))
        .applicationVersion(VK_MAKE_API_VERSION(0, 0, 1, 0))
        .pEngineName(MemoryStack.stackUTF8("coreEngine.vk"))
        .engineVersion(VK_MAKE_API_VERSION(0, 0, 1, 0))
        .apiVersion(VK_API_VERSION_1_3);

      var instanceInfo = VkInstanceCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
        .pApplicationInfo(appInfo)
        .ppEnabledExtensionNames(extensionsPtr)
        .ppEnabledLayerNames(layersPtr)
        .pNext(debug ? debugInfo.address() : MemoryUtil.NULL); //pNext cannot accept a null object. If its not a long representing an address, it must be non-null
                                                               //pNext creates a linked list of info structs for the API to consume

      //creates the instance on the driver side
      var ptr = trackMem(MemoryUtil.memAllocPointer(1));
      checkResult(vkCreateInstance(instanceInfo, null, ptr));

      //creates the instance on java side, and fills out the available capabilities / extensions for lookup convenience
      instance = new VkInstance(ptr.get(), instanceInfo);

      if(debug){
        var dbgPtr = stack.mallocLong(1);//trackMem(MemoryUtil.memAllocLong(1));
        vkCreateDebugUtilsMessengerEXT(instance, debugInfo, null, dbgPtr);
        debugMessenger = dbgPtr.get();

        System.out.println("Hooked in debug messenger");
      }
    }
  }

  private void createSurface(){
    //abstracts away interacting with the platform specific vulkan extensions to create a drawing surface, but they all work similar, with a createInfo object and an API call to create a surface
    //Thats why the call returns a vulkan result to check against

    try(var stack = MemoryStack.stackPush()){
      var ptr = stack.mallocLong(1);//trackMem(MemoryUtil.memAllocLong(1));
      checkResult(glfwCreateWindowSurface(instance, window, null, ptr));
      surface = ptr.get();
    }

    System.out.println("Created drawing surface");
  }

  private void selectPhyscialDevice(){
    try(var stack = MemoryStack.stackPush()){
      var numPtr = stack.mallocInt(1);
      vkEnumeratePhysicalDevices(instance, numPtr, null);

      if(numPtr.get(0) == 0) throw new RuntimeException("Found no vulkan capable devices");

      var devicePtr = stack.mallocPointer(numPtr.get(0));
      vkEnumeratePhysicalDevices(instance, numPtr, devicePtr);

      List<DeviceInfo> devices = new ArrayList<>();

      while(devicePtr.remaining() > 0){
        //VkPhysicalDevices are managed by the API, no need to free
        devices.add(scoreDevice(new VkPhysicalDevice(devicePtr.get(), instance)));
      }

      devices.sort((lhs, rhs) -> rhs.score-lhs.score); //inverted; largest first

      if(devices.get(0).score == 0) throw new RuntimeException("No suitable device found");

      System.out.println("Best suited device: " + devices.get(0).deviceName);
      deviceInfo = devices.get(0);
    }
  }

  private DeviceInfo scoreDevice(VkPhysicalDevice someDevice){
    var stack      = MemoryStack.stackGet();
    var properties = VkPhysicalDeviceProperties.calloc(stack); //basic info, such as device name, vencorID, but also the type (integrated, discrete, etc) and info on all the min/max limits of the API
    var features   = VkPhysicalDeviceFeatures  .calloc(stack); //more detailed info on features, such as texture compression, 64bit float support, etc.

    vkGetPhysicalDeviceProperties(someDevice, properties);
    vkGetPhysicalDeviceFeatures  (someDevice, features  );

    var name   = properties.deviceNameString();
    var type   = properties.deviceType();
    var queues = queryQueueFamilyInfo(someDevice);

    //requred
    var hasGeoShader  = features.geometryShader()     ? 1 : 0;
    var hasGraphics   = queues.graphics != null       ? 1 : 0;
    var hasPresent    = queues.present  != null       ? 1 : 0;
    var hasCompute    = queues.compute  != null       ? 1 : 0;
    var hasExtensions = checkDeviceExtensions(someDevice) ? 1 : 0;
    var hasSwapChain  = 0;

    SwapChainConfig deviceSwapChain = null;
    if(hasExtensions > 0){ //swapchain query only possible if the swapchain extension is present
      deviceSwapChain = querySwapChainConfig(someDevice);

      hasSwapChain = (deviceSwapChain.formats.length > 0 && deviceSwapChain.presentModes.length > 0) ? 1 : 0;
    }

    //preferred
    var discrete = type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 : 1;
    var hasTransfer = queues.transfer != null ? 2 : 1;

    int score = (discrete + hasTransfer) * (hasGeoShader * hasGraphics * hasPresent * hasCompute * hasExtensions * hasSwapChain);

    return new DeviceInfo(score, someDevice, queues, deviceSwapChain, name);
  }

  private QueueFamilyConfig queryQueueFamilyInfo(VkPhysicalDevice dev){
    var stack = MemoryStack.stackGet();

    var numPtr = stack.mallocInt(1);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, numPtr, null);
    var queues = VkQueueFamilyProperties.calloc(numPtr.get(0), stack);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, numPtr, queues);

    Integer graphics = null;
    Integer present  = null;
    Integer compute  = null;
    Integer transfer = null;

    for(int i = 0; i < queues.capacity(); i++){
      var queue = queues.get(i);
      int flags = queue.queueFlags();

      if ((flags & VK_QUEUE_GRAPHICS_BIT) != 0) graphics = i;
      if ((flags & VK_QUEUE_COMPUTE_BIT)  != 0) compute  = i;
      if ((flags & VK_QUEUE_TRANSFER_BIT) != 0) transfer = i;

      var ptr = stack.mallocInt(1);
      vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, ptr);

      if(ptr.get() == VK_TRUE) present = i;
    }

    return new QueueFamilyConfig(graphics, present, compute, transfer);
  }

  private SwapChainConfig querySwapChainConfig(VkPhysicalDevice someDevice){
    try(var stack = MemoryStack.stackPush()){
      var capabilities = trackMem(VkSurfaceCapabilitiesKHR.calloc());
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(someDevice, surface, capabilities);

      var cntPtr = stack.mallocInt(1);
      vkGetPhysicalDeviceSurfaceFormatsKHR(someDevice, surface, cntPtr, null);
      var formatPtr = trackMem(VkSurfaceFormatKHR.calloc(cntPtr.get(0)));
      vkGetPhysicalDeviceSurfaceFormatsKHR(someDevice, surface, cntPtr, formatPtr);

      vkGetPhysicalDeviceSurfacePresentModesKHR(someDevice, surface, cntPtr, null);
      var presentPtr = stack.mallocInt(cntPtr.get(0));
      vkGetPhysicalDeviceSurfacePresentModesKHR(someDevice, surface, cntPtr, presentPtr);

      VkSurfaceFormatKHR[] formats = new VkSurfaceFormatKHR[formatPtr.capacity()];
      int[] presentModes = new int[presentPtr.capacity()];

      int i = 0; while(formatPtr .remaining() > 0) formats     [i++] = formatPtr .get();
          i = 0; while(presentPtr.remaining() > 0) presentModes[i++] = presentPtr.get();

      return new SwapChainConfig(capabilities, formats, presentModes);
    }
  }

  private boolean checkDeviceExtensions(VkPhysicalDevice someDevice){
    try(var stack = MemoryStack.stackPush()){
      var cntPtr = stack.mallocInt(1);
      vkEnumerateDeviceExtensionProperties(someDevice, (ByteBuffer)null, cntPtr, null);
      var properties = VkExtensionProperties.calloc(cntPtr.get(0), stack);
      vkEnumerateDeviceExtensionProperties(someDevice, (ByteBuffer)null, cntPtr, properties);

      var checkSet = new HashSet<>(requiredDeviceExtensions);

      for(var prop : properties){
        checkSet.remove(prop.extensionNameString());
      }

      return checkSet.isEmpty();
    }
  }

  //output format settings & srgb handling
  private VkSurfaceFormatKHR chooseSwapSurfaceFormat(VkSurfaceFormatKHR[] formats){
    for(var format : formats){
      if(format.format() == VK_FORMAT_R8G8B8A8_SRGB && format.colorSpace() == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return format;
    }

    return formats[0];
  }

  //vsync settings
  private int chooseSwapPresentMode(int[] modes){
    int fallback = modes[0];

    Set<Integer> modeSet = new HashSet<>();
    for(int mode : modes) modeSet.add(mode);

    return modeSet.contains(VK_PRESENT_MODE_MAILBOX_KHR)      ? VK_PRESENT_MODE_MAILBOX_KHR :      //tripple buffered
           modeSet.contains(VK_PRESENT_MODE_FIFO_RELAXED_KHR) ? VK_PRESENT_MODE_FIFO_RELAXED_KHR : //vsync, but if a frame is late acceppt tearing
           modeSet.contains(VK_PRESENT_MODE_FIFO_KHR)         ? VK_PRESENT_MODE_FIFO_KHR :         //full vsync
           fallback; //probably no sync at all
  }

  //size of the "default framebuffer"
  private VkExtent2D chooseSwapExtend(VkSurfaceCapabilitiesKHR capabilities){
    //-1 indicates that the extent can be chosen, otherwise the provided extend has to be used
    if(capabilities.currentExtent().width() != -1) return capabilities.currentExtent();

    System.out.println("Setting up custom extent");

    var stack = MemoryStack.stackGet();
    var wPtr = stack.mallocInt(1);
    var hPtr = stack.mallocInt(1);

    glfwGetFramebufferSize(window, wPtr, hPtr);

    int w = Math.max(capabilities.minImageExtent().width(),  Math.min(capabilities.maxImageExtent().width(),  wPtr.get()));
    int h = Math.max(capabilities.minImageExtent().height(), Math.min(capabilities.maxImageExtent().height(), hPtr.get()));

    var actualExtent = VkExtent2D.create();
    actualExtent.set(w, h);

    return actualExtent;
  }

  private void createLogicalDevice(){
    try(var stack = MemoryStack.stackPush()){

      //some queue families in the below set might have the same index. This means they are the same family internally, and the queue creation doesn't allow multiple creations of the same family (except via queueCount)
      //Java-side the queues are separated logically, but some queue families might support the features of multiple logical queues, therefore pointing to the same family index
      Set<Integer> uniqueFamilies = new HashSet<>();
      uniqueFamilies.add(deviceInfo.queues.graphics);
      uniqueFamilies.add(deviceInfo.queues.present);
      uniqueFamilies.add(deviceInfo.queues.compute);
      uniqueFamilies.add(deviceInfo.queues.transfer);

      var queueInfos = VkDeviceQueueCreateInfo.calloc(uniqueFamilies.size(), stack);
      int q_idx = 0;
      var priority = stack.floats(1);

      for(var idx : uniqueFamilies){
        queueInfos.get(q_idx++)
        .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
        .queueFamilyIndex(idx)
        //.queueCount(1);            //doesn't exist in the java version
        .pQueuePriorities(priority); //automatically sets the queue count based on the amount of priorities
      }

      var deviceFeatures = VkPhysicalDeviceFeatures.calloc(stack); //no features requested. TODO for later
      var deviceExtensions = getDeviceExtensionsPtr();

      var deviceCreateInfo = VkDeviceCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
        .pQueueCreateInfos(queueInfos)
        .pEnabledFeatures(deviceFeatures)
        .ppEnabledExtensionNames(deviceExtensions);

      var logicalDevicePtr = stack.mallocPointer(1);
      checkResult(vkCreateDevice(deviceInfo.device, deviceCreateInfo, null, logicalDevicePtr));

      device = new VkDevice(logicalDevicePtr.get(), deviceInfo.device, deviceCreateInfo);

      var queuePtr = stack.mallocPointer(1);
      vkGetDeviceQueue(device, deviceInfo.queues.graphics, 0, queuePtr); queueGraphics = new VkQueue(queuePtr.get(0), device);
      vkGetDeviceQueue(device, deviceInfo.queues.present,  0, queuePtr); queuePresent  = new VkQueue(queuePtr.get(0), device);
      //vkGetDeviceQueue(device, deviceInfo.queues.compute , 0, queuePtr); queueCompute  = new VkQueue(queuePtr.get(0), device);
      //vkGetDeviceQueue(device, deviceInfo.queues.transfer, 0, queuePtr); queueTransfer = new VkQueue(queuePtr.get(0), device);

      System.out.println("Created logical device with " + queueInfos.capacity() + " unique, " + 4 + " logical queues");
    }
  }

  private void createSwapChain(){
    try(var stack = MemoryStack.stackPush()){
      var swapChainCfg = deviceInfo.swapchain;

      var surfaceFormat = chooseSwapSurfaceFormat(swapChainCfg.formats     );
      var presentMode   = chooseSwapPresentMode  (swapChainCfg.presentModes);
      var extent        = chooseSwapExtend       (swapChainCfg.capabilities);

      var chainLength   = swapChainCfg.capabilities.minImageCount() + 1; //only tripple buffering is flexible with their length, where min == 2, so +1 is reasonable
      if(swapChainCfg.capabilities.maxImageCount() != 0) chainLength = Math.min(swapChainCfg.capabilities.maxImageCount(), chainLength);

      var createInfo = VkSwapchainCreateInfoKHR.calloc(stack)
        .sType            (VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR)
        .surface          (surface)
        .minImageCount    (chainLength)
        .imageFormat      (surfaceFormat.format())
        .imageColorSpace  (surfaceFormat.colorSpace())
        .imageExtent      (extent)
        .presentMode      (presentMode)
        .imageArrayLayers (1)                                         //non-1 for stereoscopic rendering
        .imageUsage       (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)       //for direct rendering. With postprocessing theres the option for VK_IMAGE_USAGE_TRANSFER_DST_BIT and doing a blit operation
        .preTransform     (swapChainCfg.capabilities.currentTransform()) //allows preTransformations, for things such as mobile screens that rotate by 90° and such
        .compositeAlpha   (VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)         //can use alpha blending on the surface, deactivated for now
        .clipped          (true)                                      //means pixels that are obscured by the OS windowing system are irelevant. As read-back from the swapchain isn't a usecase here, just clip it
        .oldSwapchain     (MemoryUtil.NULL);                          //on window resize, swapchains need to be re-created, in such a case a pointer to the old chain must be supplied

      //if the swapchain interacts with multiple queues, there needs to be a policy how synchronization is done
      if(deviceInfo.queues.graphics != deviceInfo.queues.present){
        var idxPtr = stack.ints(deviceInfo.queues.graphics, deviceInfo.queues.present);
        createInfo.imageSharingMode(VK_SHARING_MODE_CONCURRENT)
                  .pQueueFamilyIndices(idxPtr);
      }else{
        createInfo.imageSharingMode(VK_SHARING_MODE_EXCLUSIVE);
      }

      var ptr = stack.mallocLong(1);
      vkCreateSwapchainKHR(device, createInfo, null, ptr);
      swapChain = ptr.get();

      var cntPtr = new int[1];//stack.mallocInt(1);
      vkGetSwapchainImagesKHR(device, swapChain, cntPtr, null);
      swapChainImages = new long[cntPtr[0]];
      vkGetSwapchainImagesKHR(device, swapChain, cntPtr, swapChainImages); //retrieves the handles to the VkImages (Textures in GL)

      deviceInfoExtra = new SwapChainExtra(surfaceFormat, extent);

      System.out.println("Swapchain info: format = " + surfaceFormat + ", mode = " + presentMode + ", extent = (" + extent.width() + "x" + extent.height() + "), requested length = " + chainLength + ", provided length = " + swapChainImages.length);
    }
  }

  private void createImageViews(){
    //VkImages are the handles for the textures storage. Unlike in GL, accessing the storage requires a VkImageView (wheras GLs TextureViews were optional)

    swapChainImageViews = new long[swapChainImages.length];

    int i = 0;
    for(var image : swapChainImages){
      try(var stack = MemoryStack.stackPush()){
        var createInfo = VkImageViewCreateInfo.calloc(stack)
          .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
          .image(image)
          .viewType(VK_IMAGE_VIEW_TYPE_2D) //view the texture as a different type, such as viewing at a 2D slice of a 2Darray
          .format(deviceInfoExtra.usedFormat.format()) //views can have different formats, as long as they're compatible
          .components(VkComponentMapping.calloc(stack).r(VK_COMPONENT_SWIZZLE_IDENTITY)  //components in the view can be statically swizzled
                                                      .r(VK_COMPONENT_SWIZZLE_IDENTITY)
                                                      .r(VK_COMPONENT_SWIZZLE_IDENTITY)
                                                      .r(VK_COMPONENT_SWIZZLE_IDENTITY))
          .subresourceRange(VkImageSubresourceRange.calloc(stack).aspectMask(VK_IMAGE_ASPECT_COLOR_BIT) //define what usage type the texture will have (color, depth, etc) as well as what region to view
                                                                 .baseMipLevel(0)
                                                                 .levelCount(1)
                                                                 .baseArrayLayer(0)
                                                                 .layerCount(1));
        var ptr = stack.mallocLong(1);
        vkCreateImageView(device, createInfo, null, ptr);
        swapChainImageViews[i++] = ptr.get(); //while the images in the swapchain are owned by the swapchain, these views need explicit cleanup
      }
    }
  }

  private ByteBuffer compileShaderStage(String path, int type){
    String fn = path.substring(path.lastIndexOf("/")+1, path.length());

    try{
      String src = Files.readString(Paths.get(path));
      long ptr = Shaderc.shaderc_compile_into_spv(shaderCompiler, src, type, fn, "main", compilerOptions);

      var state = Shaderc.shaderc_result_get_compilation_status(ptr);
      if(state != Shaderc.shaderc_compilation_status_success){
        String error = Shaderc.shaderc_result_get_error_message(ptr);
        throw new RuntimeException(error);
      }

      var data = Shaderc.shaderc_result_get_bytes(ptr);
      //shaderc talsk about guaranteed uint32_t alignment, so use that to be sure
      var result = trackMem(MemoryUtil.memAlignedAlloc(32, data.capacity()));
      result.put(data);
      Shaderc.shaderc_result_release(ptr);

      return result.rewind();
    } catch (IOException x){
      throw new RuntimeException(x);
    }
  }

  private long createShaderModule(ByteBuffer stage){
    var stack = MemoryStack.stackGet();

    var createInfo = VkShaderModuleCreateInfo.calloc(stack)
      .sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
      .pCode(stage);

    var ptr = stack.mallocLong(1);
    checkResult(vkCreateShaderModule(device, createInfo, null, ptr));

    return ptr.get();
  }

  private void createRenderPass(){
    //apparently this is optional, and can be skipped by an extension
    //those passes are supposedly only useful for mobile devices tile rendering

    //The closes analogy to a renderPass is a coreEngine pipeline stage
    //Vulkan render passes can (must?) have subpasses, so this means at least 1

    //The render pass is, in Vulklan, kind of an interface between the Framebuffer (VkImages), and the pipeline object

    try(var stack = MemoryStack.stackPush()){
      var attachInfo = VkAttachmentDescription.calloc(1, stack)
        .format        (deviceInfoExtra.usedFormat.format())
        .samples       (VK_SAMPLE_COUNT_1_BIT           )
        .loadOp        (VK_ATTACHMENT_LOAD_OP_CLEAR     ) //its handy to specify what happens for the load. don't care, clear, or load previous contents
        .storeOp       (VK_ATTACHMENT_STORE_OP_STORE    ) //can also specify don't care
        .stencilLoadOp (VK_ATTACHMENT_LOAD_OP_DONT_CARE )
        .stencilStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE)
        .initialLayout (VK_IMAGE_LAYOUT_UNDEFINED       )  //the internal layout of VkImages on the GPU may chaneg depending on the use cases. UNDEFINED because its going to be cleared anyway
        .finalLayout   (VK_IMAGE_LAYOUT_PRESENT_SRC_KHR ); //as the image is pushed straight to the swapchain after the triangle, put it into the optimal present mode


      //each subpass references one or more attachmentRefs, and each attachment is in the end an VkImage, so they'll have the referenced layout. Only after all subpasses finished will the image transition
      //to the finalLayout of the overall renderPass
      var attachRefs = VkAttachmentReference.calloc(1, stack)
        .attachment(0)
        .layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

      var subpassInfos = VkSubpassDescription.calloc(1, stack)
        .pipelineBindPoint(VK_PIPELINE_BIND_POINT_GRAPHICS) //graphics and compute are the 2 possible bind points
        .colorAttachmentCount(1) //mind the lwjgl API inconsistency
        .pColorAttachments(attachRefs);
        //there are other types of attachments, the most usual DepthStencil, but there's also resolve for multisampling, and input for "read from a shader" (?) maybe buffer textures?

      //specifies dependencies between subpasses (where begin/end also have to be managed)
      //what's unclear is why that's needed, as all dependenvies are already handled by the cmdBuffer recorded. Maybe this is only for the tutorial?
      var subPassDependencies = VkSubpassDependency.calloc(1, stack)
        .srcSubpass(VK_SUBPASS_EXTERNAL) //src==external means this is the 1st pass, dst==external means this would be the last pass
        .dstSubpass(0)
        .srcStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT) //isn't that already done with the cmdBuffer semaphore wait?
        .srcAccessMask(0)                                            //
        .dstStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT) //
        .dstAccessMask(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);        //

      var renderPassCreateInfo = VkRenderPassCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO)
        .pAttachments(attachInfo)
        .pSubpasses(subpassInfos)
        .pDependencies(subPassDependencies);



      var ptr = stack.mallocLong(1);
      checkResult(vkCreateRenderPass(device, renderPassCreateInfo, null, ptr));
      renderPass = ptr.get();

      System.out.println("Render pass object created successfully");
    }
  }

  private void createPipeline(){
    //ideas on uniforms:

    //for "static" data (per material) use a per-material ubo (each material binds it's ubo to the shader's material-ubo slot)
    //for small data, use push constants / push descriptors (which is then bound to a object-ubo slot in the shader)
    //for large data, use a per-frame approach. So either the data is used once per frame (e.g. particles),
    //or it can be aggregated into a frame-cache (e.g. cached matrices)

    //small per object data lives in the cmd buffer, so that is easy peasy lemon sqeezy
    //large buffers track multiple host-local staging buffers, with the data provided for upload()/update()
    //this gets mem-cpyd to a device-local buffer, which is the _actual_ ubo/ssbo.
    //semaphores are used to determine when the memcpy can be performed, and signaled fences tell the host
    //when old staging buffers can be destroyed / reused.

    //static data should just be uploaded once, perhaps that too requires an ugly staging buffer. But maybe when done once, it can reuse the large-buffer staging-buffer tracking logic
    //and clean up its only staging buffer

    //what about the g_data ubo. How much is actually frame-static?
    // -> not much. framebuffer sizes change all the time. cameras change at least once
    // -> g_lighting would be frame-static
    // -> g_lineData is more like a per-object thing
    //    -> how does all of this relate to UI drawing? Probably all of it into push descriptors?
    // -> A: All of them are < 512 bytes, this should fit into a push descriptor

    //this way the API should stay mostly the same (perhaps a finishedUse() might be necessary to set the proper semaphore?)
    //and all the annoying data is host-local (and perhaps the device-cpy can be skipped, if it can be host_local | device_visible, but that would need profiling?)




    //how to handle CommandBuffers? At the start of a frame (CPU side) the cmdBuffers might still very well be in-flight. So either have completely static cmdBuffers that can be re-queued as-is,
    //or have fire-and-forget cmdBuffers, and re-record them for every object drawn
    //in addition, this could mean either a buffer per object, or a buffer per subpass

    try(var stack = MemoryStack.stackPush()){
      //using ShaderC for cpmpilation
      var vert = compileShaderStage("assets/shaders/vert.vsh", Shaderc.shaderc_vertex_shader);
      var frag = compileShaderStage("assets/shaders/frag.fsh", Shaderc.shaderc_fragment_shader);

      //creates a vulkan-known wrapper around the spv bytecode
      vertModule = createShaderModule(vert);
      fragModule = createShaderModule(frag);

      var stagesCreateInfo = VkPipelineShaderStageCreateInfo.calloc(2, stack);
      var entryPoint = stack.UTF8("main");

      stagesCreateInfo.get(0)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
        .stage(VK_SHADER_STAGE_VERTEX_BIT)
        .module(vertModule)
        .pName(entryPoint);

      stagesCreateInfo.get(1)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
        .stage(VK_SHADER_STAGE_FRAGMENT_BIT)
        .module(fragModule)
        .pName(entryPoint);

      var statePtr = stack.ints(VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR);
      var dynStateCreateInfo = VkPipelineDynamicStateCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO)
        .pDynamicStates(statePtr);

      //vertex layout is actually part of the pipeline/shader, not the vertex buffer
      var vertInputCreateInfo = VkPipelineVertexInputStateCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
        //.pVertexBindingDescriptions(null)
        //.pVertexAttributeDescriptions(null);

      //what was the old "geoMode" is also part of the pipeline/shader
      var vertAssemblyCreateInfo = VkPipelineInputAssemblyStateCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
        .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .primitiveRestartEnable(false); //can be part of dynamic state

      var viewportCreateInfo = VkPipelineViewportStateCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
        .viewportCount(1)
        .scissorCount(1); //would imply multiply scissor regions, cool

      var rasterizerCreateInfo = VkPipelineRasterizationStateCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
        .depthClampEnable(false)           //could clamp depth results to stay inside the frustum, usefuly for lights?
        .rasterizerDiscardEnable(false)    //discard all fragment writes if enabled
        .polygonMode(VK_POLYGON_MODE_FILL) //tri, line, point
        .lineWidth(1)
        .cullMode(VK_CULL_MODE_BACK_BIT)   //cullmode is done in the pipeline. can by dynamic too. Also its front/back, and then also telling the pipeline of ccw/cw is from or back
        .frontFace(VK_FRONT_FACE_CLOCKWISE)//travesty, yes. But following the tutorial
        .depthBiasEnable(false);           //also optionally dynamic

      //noone cares about multisample? :|
      var multisampleCreateInfo = VkPipelineMultisampleStateCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
        .sampleShadingEnable(false)
        .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT);
        //.minSampleShading(1)
        //.pSampleMask(null)
        //.alphaToCoverageEnable(false)
        //.alphaToOneEnable(false);

/*       var depthCreateInfo = VkPipelineDepthStencilStateCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO)
        .depthBoundsTestEnable(false)
        .depthCompareOp(VK_COMPARE_OP_ALWAYS)
        .depthTestEnable(false)
        .depthWriteEnable(false); */

      //blendModes are also pipeline bound, one for each framebuffer attachment
      var blendCreateInfo = VkPipelineColorBlendAttachmentState.calloc(1, stack)
        .colorWriteMask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT) //can selectivley enable which channels get written
        .blendEnable(false);
        //.srcColorBlendFactor(VK_BLEND_FACTOR_ONE)
        //.dstColorBlendFactor(VK_BLEND_FACTOR_ZERO)
        //.colorBlendOp(VK_BLEND_OP_ADD)
        //.srcAlphaBlendFactor(VK_BLEND_FACTOR_ONE)
        //.dstAlphaBlendFactor(VK_BLEND_FACTOR_ZERO)
        //.alphaBlendOp(VK_BLEND_OP_ADD);

      //in addition there is some bit-based logic blending possible (?)
      //but otherwise the state holds the blendInfos for alle the attachments just created. What the blendConstants are for is not clear
      var blendStateCreateInfo = VkPipelineColorBlendStateCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO)
        .logicOpEnable(false)
        .logicOp(VK_LOGIC_OP_COPY)
        .pAttachments(blendCreateInfo)
        .blendConstants(stack.floats(0,0,0,0));

      //the pipeline also specifies what uniforms will be available
      var layoutCreateInfo = VkPipelineLayoutCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
        //.pSetLayouts(null)
        //.pPushConstantRanges(null);

      var layoutPtr = stack.mallocLong(1);
      checkResult(vkCreatePipelineLayout(device, layoutCreateInfo, null, layoutPtr));
      pipelineLayout = layoutPtr.get();

      var pipelinecreateInfo = VkGraphicsPipelineCreateInfo.calloc(1, stack)
        .sType              (VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
        .pStages            (stagesCreateInfo      )
        .pVertexInputState  (vertInputCreateInfo   )
        .pInputAssemblyState(vertAssemblyCreateInfo)
        .pViewportState     (viewportCreateInfo    )
        .pRasterizationState(rasterizerCreateInfo  )
        .pMultisampleState  (multisampleCreateInfo )
        //.pDepthStencilState (depthCreateInfo       ) //TODO
        .pColorBlendState   (blendStateCreateInfo  )
        .pDynamicState      (dynStateCreateInfo    )
        .layout             (pipelineLayout        )
        .renderPass         (renderPass            )
        .subpass            (0                     )  //this uses the 1st subpass
        .basePipelineHandle (VK_NULL_HANDLE        )  //possible to derive from already existing pipelines
        .basePipelineIndex  (-1                    ); //could also reference an index in this buffer of createInfos for derivation

      //instead of MemoryUtil.NULL, it is possible to supply a pipeline cache, i.e. a ShaderCache
      var pipelinePtr = stack.mallocLong(1);
      checkResult(vkCreateGraphicsPipelines(device, MemoryUtil.NULL, pipelinecreateInfo, null, pipelinePtr));
      pipeline = pipelinePtr.get();

      System.out.println("Pipeline created");
    }
  }

  private void createFramebuffers(){
    try(var stack = MemoryStack.stackPush()){
      //framebuffers are a bit thinner than their openGL counterpart

      swapChainFramebuffers = new long[swapChainImageViews.length];

      var i = 0;
      var ptr = stack.mallocLong(1);
      for(var imgView : swapChainImageViews){
        var createinfo = VkFramebufferCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO)
        .renderPass(renderPass)
        .attachmentCount(1)
        .pAttachments(stack.longs(imgView))
        .width(deviceInfoExtra.usedExtent.width())
        .height(deviceInfoExtra.usedExtent.height())
        .layers(1); //layers in the imgView. AS its just of type 2D, layers=1

        checkResult(vkCreateFramebuffer(device, createinfo, null, ptr));
        swapChainFramebuffers[i++] = ptr.get(0);
      }
    }
  }

  private void createCommandPool(){
    try(var stack = MemoryStack.stackPush()){
      var createinfo = VkCommandPoolCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
        .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) //allows existing cmdBuffers to be individually rewritten (?) there is also  VK_COMMAND_POOL_CREATE_TRANSIENT_BIT which hints that a buffer is rewritten very often (?)
        .queueFamilyIndex(deviceInfo.queues.graphics);

      var ptr = stack.mallocLong(1);
      checkResult(vkCreateCommandPool(device, createinfo, null, ptr));
      commandPool = ptr.get();
    }
  }

  private void createCommandBuffer(){
    try(var stack = MemoryStack.stackPush()){
      var createinfo = VkCommandBufferAllocateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
        .commandPool(commandPool)
        .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY) //there are also secondary cmdBuffers, but they can only be invoked by primarey cmdBuffers
        .commandBufferCount(1);

      var ptr = stack.mallocPointer(1);
      checkResult(vkAllocateCommandBuffers(device, createinfo, ptr));
      cmdBuffer = new VkCommandBuffer(ptr.get(), device);
    }
  }

  private void createSyncObjects(){
    try(var stack = MemoryStack.stackPush()){
      var semaInfo = VkSemaphoreCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);

      var fenceInfo = VkFenceCreateInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        .flags(VK_FENCE_CREATE_SIGNALED_BIT); //the loop is setup in a way that the 1st operation will be a wait, so instance the fence already signaled

      var semPtr1 = stack.mallocLong(1);
      var semPtr2 = stack.mallocLong(1);
      var fenPtr  = stack.mallocLong(1);

      vkCreateSemaphore(device, semaInfo,  null, semPtr1);
      vkCreateSemaphore(device, semaInfo,  null, semPtr2);
      vkCreateFence    (device, fenceInfo, null, fenPtr);

      imgAvailableSem   = semPtr1.get();
      renderFinishedSem = semPtr2.get();
      inFlightFence     = fenPtr.get();
    }
  }

  private void checkResult(int code){
    //TODO extend to all VK error codes. There is a description/list of all codes in the javadoc / vulkan spec
    String error = switch (code){
      case VK_SUCCESS                     -> null;
      case VK_ERROR_OUT_OF_HOST_MEMORY    -> "VK_ERROR_OUT_OF_HOST_MEMORY";
      case VK_ERROR_OUT_OF_DEVICE_MEMORY  -> "ERROR_OUT_OF_DEVICE_MEMORY";
      case VK_ERROR_INITIALIZATION_FAILED -> "ERROR_INITIALIZATION_FAILED";
      case VK_ERROR_LAYER_NOT_PRESENT     -> "ERROR_LAYER_NOT_PRESENT";
      case VK_ERROR_EXTENSION_NOT_PRESENT -> "ERROR_EXTENSION_NOT_PRESENT";
      case VK_ERROR_INCOMPATIBLE_DRIVER   -> "ERROR_INCOMPATIBLE_DRIVER";
      case VK_ERROR_DEVICE_LOST           -> "VK_ERROR_DEVICE_LOST";
      case VK_ERROR_OUT_OF_DATE_KHR       -> "VK_ERROR_OUT_OF_DATE_KHR";
      case VK_ERROR_SURFACE_LOST_KHR      -> "VK_ERROR_SURFACE_LOST_KHR";

      default                             -> "Unknown error " + code;
    };

    if(error != null) throw new RuntimeException(error);
  }

  public void run(){
    System.out.println("Setup completed successfully");

    while(!glfwWindowShouldClose(window)){
      glfwPollEvents();
      drawFrame();
    }

    vkDeviceWaitIdle(device);
  }

  private void drawFrame(){
    try(var stack = MemoryStack.stackPush()){
      //makes sure the cmdBuffer was used up and can be rerecorded
      vkWaitForFences(device, inFlightFence, true, -1L);
      vkResetFences(device, stack.longs(inFlightFence));

      var idxPtr = stack.mallocInt(1);
      checkResult(vkAcquireNextImageKHR(device, swapChain, -1L, imgAvailableSem, VK_NULL_HANDLE, idxPtr)); //returns the next image from the chain, and signals the semaphore as soon as the returned
                                                                                                           //image is no longer needed by the swapchain. It returns the index in the swapchain, the
                                                                                                           //corresponding image/view has to be retrieved afterwards
      var imageIndex = idxPtr.get();
      recordCommandBuffer(cmdBuffer, imageIndex);

      var submitInfo = VkSubmitInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
        .waitSemaphoreCount(1)                         //mind the lwjgl API inconsistency
        .pWaitSemaphores(stack.longs(imgAvailableSem)) //wait (after then reset) on the following semaphores (which will be when the swapchain says its no longer needed)
        .pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)) //the semaphore wait isn't binary, but can be for certain stages. Currently the img just have to be ready to be an color attachment
        .pCommandBuffers(stack.pointers(cmdBuffer.address()))
        .pSignalSemaphores(stack.longs(renderFinishedSem)); //semaphores to signal

      var presentInfo = VkPresentInfoKHR.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR)
        .pWaitSemaphores(stack.longs(renderFinishedSem)) //wait on those semaphores before presenting
        .swapchainCount(1)                               //mind the lwjgl API inconsistency
        .pSwapchains(stack.longs(swapChain))             //the chain to present to
        .pImageIndices(stack.ints(imageIndex));          //the index in the chain to present

      checkResult(vkQueueSubmit(queueGraphics, submitInfo, inFlightFence)); //signals the fence when the cmdBuffer has finished executing
      checkResult(vkQueuePresentKHR(queuePresent, presentInfo));
    }

  }

  private void recordCommandBuffer(VkCommandBuffer useBuffer, int target){
    try(var stack = MemoryStack.stackPush()){

      /* possible flags:
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT:      The command buffer will be rerecorded right after executing it once.
        VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: This is a secondary command buffer that will be entirely within a single render pass.
        VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT:     The command buffer can be resubmitted while it is also already pending execution.
      */

      var cmdBeginInfo = VkCommandBufferBeginInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
        .flags(0)
        .pInheritanceInfo(null);

      var passBeginInfo = VkRenderPassBeginInfo.calloc(stack)
        .sType(VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO)
        .renderPass(renderPass)
        .framebuffer(swapChainFramebuffers[target])
        .renderArea(VkRect2D.calloc(stack)
          .offset(VkOffset2D.calloc(stack)
            .x(0)
            .y(0))
          .extent(deviceInfoExtra.usedExtent))
        .pClearValues(VkClearValue.calloc(1, stack)
          .color(VkClearColorValue.calloc(stack)
            .uint32(stack.ints(0))));

      //use the dx way of mapping depth
      var viewports = VkViewport.calloc(1, stack)
        .x(0)
        .y(9)
        .width(deviceInfoExtra.usedExtent.width())
        .height(deviceInfoExtra.usedExtent.height())
        .minDepth(1)
        .maxDepth(0);

      var scissors = VkRect2D.calloc(1, stack)
        .offset(VkOffset2D.calloc(stack)
          .x(0)
          .y(0))
        .extent(deviceInfoExtra.usedExtent);

      checkResult(vkResetCommandBuffer(useBuffer, 0)); //fencing should make sure this is done only after the buffer was executed entirely
      checkResult(vkBeginCommandBuffer(useBuffer, cmdBeginInfo));

      //Framebuffer.BINDING.bind(...)
      vkCmdBeginRenderPass(useBuffer, passBeginInfo, VK_SUBPASS_CONTENTS_INLINE); //vkCmd... returns void, the result can be checked when the buffer is submitted
                                                                                  //VK_SUBPASS_CONTENTS_INLINE menas no usage of secondary cmdBuffers (?)
      //material.bind(...)
      vkCmdBindPipeline(useBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

      vkCmdSetViewport(useBuffer, 0, viewports);
      vkCmdSetScissor (useBuffer, 0, scissors );

      //mesh.draw(...)
      //this works, as while there is no vertex buffer, 3 vertices are emitted, gl_VertexIndex will be advanced, and the position read from the static array in the vertex shader
      vkCmdDraw(useBuffer, 3, 1, 0, 0);

      vkCmdEndRenderPass(useBuffer);
      checkResult(vkEndCommandBuffer(useBuffer));
    }
  }

  public void dispose(){
    disposeVulkan();
    disposeGlfw();
    disposeShaderC();
    freeMemory();
  }

  private void disposeVulkan(){
    System.out.println("Disposing vulkan specifics");

    vkDestroyFence(device, inFlightFence, null);
    vkDestroySemaphore(device, imgAvailableSem, null);
    vkDestroySemaphore(device, renderFinishedSem, null);

    vkDestroyCommandPool(device, commandPool, null);
    for(var fbo : swapChainFramebuffers) vkDestroyFramebuffer(device, fbo, null);
    vkDestroyPipeline(device, pipeline, null);
    vkDestroyPipelineLayout(device, pipelineLayout, null);
    vkDestroyRenderPass(device, renderPass, null);
    vkDestroyShaderModule(device, vertModule, null);
    vkDestroyShaderModule(device, fragModule, null);

    for(var view : swapChainImageViews) vkDestroyImageView(device, view, null);

    vkDestroySwapchainKHR(device, swapChain, null);
    vkDestroyDevice(device, null);

    if(debug) {
      vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
    }

    vkDestroySurfaceKHR(instance, surface, null);
    vkDestroyInstance(instance, null);

    if(debug){
      dbgCallback.free(); //as this callback is used during instance creation / deletion, it can only be freed after vkDestroyInstance
    }
  }

  private void disposeShaderC(){
    Shaderc.shaderc_compile_options_release(compilerOptions);
    Shaderc.shaderc_compiler_release(shaderCompiler);
  }

  private void disposeGlfw(){
    System.out.println("Disposing glfw window & api");
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  private void freeMemory(){
    for(var mem : ownedMemoryCustom) MemoryUtil.memFree(mem); //make sure to free the BUFFER, not the pointers in it
    for(var mem : ownedMemoryNIO   ) MemoryUtil.memFree(mem);
    for(var mem : ownedMemoryObj   ) mem.free();
  }
}
