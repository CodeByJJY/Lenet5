module lenet5_fpga (
    input clk,
    input reset,
    input [7:0] input_data,        // 8-bit quantized input pixel data
                                    //input image: width -> height sweep. But save reg vector: height -> width
    output reg [3:0] output_class, // Predicted class (0-9)
    output reg done                // Processing done signal
);

    // ----------------------------
    // Parameters (현재 구조 기준)
    // ----------------------------
    localparam IMG_SIZE     = 28;           
    localparam NUM_CLASSES  = 10;
    localparam NUM_ZERO_PAD = 2;

    // LeNet-variant geometry (28->24->12->8->4)
    localparam C1_K         = 5;          // Conv1 kernel
    localparam C1_OUT_CH    = 6;          // Conv1 out channels
    localparam C1_OUT_R     = 28;         // 
    localparam C1_OUT_C     = 28;

    localparam P1_OUT_R     = 14;         // 24/2
    localparam P1_OUT_C     = 14;
    localparam P1_OUT_CH    = C1_OUT_CH;  // 16

    localparam C2_K         = 5;          // Conv2 kernel
    localparam C2_IN_CH     = P1_OUT_CH;  // 6
    localparam C2_OUT_CH    = 16;         // Conv2 out channels
    localparam C2_OUT_R     = 10;          // 12-5+1
    localparam C2_OUT_C     = 10;

    localparam P2_OUT_R     = 5;          // 8/2
    localparam P2_OUT_C     = 5;
    localparam P2_OUT_CH    = C2_OUT_CH;  // 16

    // Flatten size (중요!)
    localparam FLAT_SIZE    = P2_OUT_R * P2_OUT_C * P2_OUT_CH; // 5*5*16=256

    // FC 차원 (이 값들에 맞춰 ROM 생성해야 함)
    localparam FC1_OUT      = 120;
    localparam FC2_OUT      = 84;
    localparam FC3_OUT      = NUM_CLASSES;
    

    // State Machine States
    localparam  IDLE                    = 0, 
                LOAD_IMAGE              = 1,   
                LOAD_INIT_CONV1_WEIGHT  = 2,
                LOAD_INIT_CONV2_WEIGHT  = 3,
                LOAD_INIT_CONV1_BIAS    = 4,
                LOAD_INIT_CONV2_BIAS    = 5,
                LOAD_INIT_FC1_BIAS      = 6,
                LOAD_INIT_FC2_BIAS      = 7,
                LOAD_INIT_FC3_BIAS      = 8,
                CONV1                   = 9, 
                POOL1                   = 10, 
                CONV2                   = 11, 
                POOL2                   = 12,
                FLATTEN                 = 13, 
                FC1                     = 14, 
                FC2                     = 15, 
                FC3                     = 16, 
                DONE                    = 17;

    localparam  SS_IDLE         =   0,  //for state_in_conv
                SS_CALL_BIAS    =   1,
                SS_GET_BIAS     =   2,
                SS_CONV         =   3,
                SS_FC           =   3,
                SS_END          =   4;
                
    reg [4:0] state;
    reg [2:0] state_in_conv;
    reg [2:0] state_in_fc;

    // Layer Buffers
    reg signed [8:0] image [0:IMG_SIZE+2*NUM_ZERO_PAD-1][0:IMG_SIZE+2*NUM_ZERO_PAD-1]; // 8-bit Image Data & zero padding	// height -> width sweep 
    reg signed [7:0] conv1_out [0:C1_OUT_CH-1][0:C1_OUT_R-1][0:C1_OUT_C-1];  // 11-bit Conv1 output
    reg signed [7:0] pool1_out [0:P1_OUT_CH-1][0:P1_OUT_R-1][0:P1_OUT_C-1];  // 11-bit Pool1 output
    reg signed [7:0] conv2_out [0:C2_OUT_CH-1][0:C2_OUT_R-1][0:C2_OUT_C-1];   // 11-bit Conv2 output
    reg signed [7:0] pool2_out [0:P2_OUT_CH-1][0:P2_OUT_R-1][0:P2_OUT_C-1];   // 11-bit Pool2 output
    reg signed [7:0] fc1_out [0:FC1_OUT-1];              // FC1 output
    reg signed [7:0] fc2_out [0:FC2_OUT-1];               // FC2 output
    reg signed [20:0] fc3_out [0:FC3_OUT-1];    // FC3 output
    
    // Weight Buffers
    reg signed [7:0]    conv1_weight[0:C1_OUT_CH-1][0:0][0:C1_K-1][0:C1_K-1];              //only sweep for input channel
    reg signed [7:0]    conv2_weight[0:C2_OUT_CH-1][0:C1_OUT_CH-1][0:C2_K-1][0:C2_K-1];    //only sweep for input channel
    reg signed [7:0]    fc1_weight[0:FLAT_SIZE-1];                       //only sweep for input nodes
    reg signed [7:0]    fc2_weight[0:FC1_OUT-1];                       //only sweep for input nodes
    reg signed [7:0]    fc3_weight[0:FC2_OUT-1];                       //only sweep for input nodes
    
    // Bias Buffers
    reg signed [7:0]    conv1_bias[0:C1_OUT_CH-1];
    reg signed [7:0]    conv2_bias[0:C2_OUT_CH-1];
    reg signed [7:0]    fc1_bias[0:FC1_OUT-1];
    reg signed [7:0]    fc2_bias[0:FC2_OUT-1];
    reg signed [7:0]    fc3_bias[0:FC3_OUT-1];

    // Flatten Buffer
    reg signed [7:0] pool2_flat [0:P2_OUT_R*P2_OUT_C*P2_OUT_CH-1]; // Flattened Pool2 Output

    // Temporary Registers
    reg signed [18:0] conv_sum; // For convolution accumulation
    reg signed [18:0] fc_sum;   // For FC accumulation
    reg signed [20:0] max_val[0:NUM_CLASSES-1];
    reg [3:0] max_index[0:NUM_CLASSES-1];
    integer i, j, k, kr, kc, c;  
    integer p, q;  
    wire [4:0] height_img, width_img;

    // Block Memory Generators for weights and biases
    wire [7:0]   dout_conv1_weight_mem,  dout_conv2_weight_mem,  dout_fc1_weight_mem,   dout_fc2_weight_mem,   dout_fc3_weight_mem;    
    wire         cen_conv1_weight_mem,   cen_conv2_weight_mem,   cen_fc1_weight_mem,    cen_fc2_weight_mem,    cen_fc3_weight_mem;

    wire [7:0] addr_conv1_weight_mem;
    wire [11:0] addr_conv2_weight_mem;
    wire [15:0] addr_fc1_weight_mem;
    wire [13:0] addr_fc2_weight_mem;
    wire [9:0] addr_fc3_weight_mem;
        
    wire [7:0]   dout_conv1_bias_mem,    dout_conv2_bias_mem,    dout_fc1_bias_mem,     dout_fc2_bias_mem,     dout_fc3_bias_mem;
    wire         cen_conv1_bias_mem,     cen_conv2_bias_mem,     cen_fc1_bias_mem,      cen_fc2_bias_mem,      cen_fc3_bias_mem;

    wire [2:0] addr_conv1_bias_mem;
    wire [3:0] addr_conv2_bias_mem;
    wire [6:0] addr_fc1_bias_mem;
    wire [6:0] addr_fc2_bias_mem;
    wire [3:0] addr_fc3_bias_mem;

    wire cen_conv1_weight_mem_buf1,  cen_conv2_weight_mem_buf1,  cen_fc1_weight_mem_buf1,   cen_fc2_weight_mem_buf1,   cen_fc3_weight_mem_buf1;
    wire cen_conv1_weight_mem_buf2,  cen_conv2_weight_mem_buf2,  cen_fc1_weight_mem_buf2,   cen_fc2_weight_mem_buf2,   cen_fc3_weight_mem_buf2;

    wire cen_conv1_bias_mem_buf1,  cen_conv2_bias_mem_buf1,  cen_fc1_bias_mem_buf1,   cen_fc2_bias_mem_buf1,   cen_fc3_bias_mem_buf1;
    wire cen_conv1_bias_mem_buf2,  cen_conv2_bias_mem_buf2,  cen_fc1_bias_mem_buf2,   cen_fc2_bias_mem_buf2,   cen_fc3_bias_mem_buf2;
   
    wire [2:0] cnt_cnn_height;
    wire [2:0] cnt_cnn_width;
    wire [3:0] cnt_cnn_in_ch;
    wire [3:0] cnt_cnn_out_ch;

    wire [31:0] cnt_init_load;
    wire [31:0] cnt_init_load_buf1;
    wire [31:0] cnt_init_load_buf2;  
    
    
    reg [4:0] cnt_cnn_if_start_width;
    reg [4:0] cnt_cnn_if_start_height;
    reg [3:0] cnt_cnn_of_ch;
    
    reg signed [7:0] curr_cnn_bias;
    reg signed [7:0] curr_cnn_filter[0:5][0:4][0:4];
    reg signed [8:0] curr_cnn_if[0:5][0:4][0:4];
    reg signed [15:0] curr_cnn_pointwise_mul[0:5][0:4][0:4];
    reg signed [22:0] curr_cnn_of_sum;
    reg signed [22:0] curr_cnn_of_sum_rounded_by_16;
    reg signed [22:0] curr_cnn_of_sum_rounded_by_16_plus_bias;
    reg signed [7:0] curr_cnn_of;    
    
    reg [4:0]   cnt_pool_if_start_width;
    reg [4:0]   cnt_pool_if_start_height;
    reg [3:0]   cnt_pool_if_ch;
    
    reg [7:0]   curr_pool_if[0:1][0:1];
    reg [9:0]   curr_pool_of_sum;
    reg [9:0]   curr_pool_of_sum_round;
    reg [7:0]   curr_pool_of;
    
    reg         pool_end;
    
    reg signed [7:0] curr_fc_bias;
    reg signed [7:0] curr_fc_weight;
    reg signed [7:0] curr_fc_if;
    reg signed [24:0] curr_fc_of_sum;
    reg signed [24:0] curr_fc_of_sum_round4;
    reg signed [24:0] curr_fc_of_sum_round4_plus_bias;
    reg signed [20:0] curr_fc_of;
    
    reg [8:0]   cnt_fc_if;
    reg [8:0]   cnt_fc_if_buf1;
    reg [8:0]   cnt_fc_if_buf2;
    reg [8:0]   cnt_fc_if_buf3;
    
    reg [6:0]   cnt_fc_of;
    
    
    
    // ================================
    // Block RAM instances (compact form)
    // ================================    
    conv1_weight_mem    conv1_weight_mem    (.clka(clk), .ena(cen_conv1_weight_mem), .wea(1'b0), .addra(addr_conv1_weight_mem), .dina(8'd0), .douta(dout_conv1_weight_mem));  
    conv2_weight_mem    conv2_weight_mem    (.clka(clk), .ena(cen_conv2_weight_mem), .wea(1'b0), .addra(addr_conv2_weight_mem), .dina(8'd0), .douta(dout_conv2_weight_mem));  
    fc1_weight_mem     fc1_weight_mem       (.clka(clk), .ena(cen_fc1_weight_mem ), .wea(1'b0), .addra(addr_fc1_weight_mem), .dina(8'd0), .douta(dout_fc1_weight_mem));    
    fc2_weight_mem     fc2_weight_mem       (.clka(clk), .ena(cen_fc2_weight_mem ), .wea(1'b0), .addra(addr_fc2_weight_mem), .dina(8'd0), .douta(dout_fc2_weight_mem));    
    fc3_weight_mem     fc3_weight_mem       (.clka(clk), .ena(cen_fc3_weight_mem ), .wea(1'b0), .addra(addr_fc3_weight_mem), .dina(8'd0), .douta(dout_fc3_weight_mem));   
     
    conv1_bias_mem  conv1_bias_mem      (.clka(clk), .ena(cen_conv1_bias_mem), .wea(1'b0), .addra(addr_conv1_bias_mem), .dina(8'd0), .douta(dout_conv1_bias_mem));  
    conv2_bias_mem  conv2_bias_mem      (.clka(clk), .ena(cen_conv2_bias_mem), .wea(1'b0), .addra(addr_conv2_bias_mem), .dina(8'd0), .douta(dout_conv2_bias_mem));    
    fc1_bias_mem   fc1_bias_mem         (.clka(clk), .ena(cen_fc1_bias_mem ), .wea(1'b0), .addra(addr_fc1_bias_mem), .dina(8'd0), .douta(dout_fc1_bias_mem));    
    fc2_bias_mem   fc2_bias_mem         (.clka(clk), .ena(cen_fc2_bias_mem ), .wea(1'b0), .addra(addr_fc2_bias_mem), .dina(8'd0), .douta(dout_fc2_bias_mem));    
    fc3_bias_mem   fc3_bias_mem         (.clka(clk), .ena(cen_fc3_bias_mem ), .wea(1'b0), .addra(addr_fc3_bias_mem), .dina(8'd0), .douta(dout_fc3_bias_mem));
    

    always @ (posedge clk or posedge reset) begin
        if(reset)   state   <=  IDLE;
        else begin
            case(state)
                IDLE:                                                   state   <=  LOAD_IMAGE;
                LOAD_IMAGE: if ((height_img == IMG_SIZE + 2*NUM_ZERO_PAD - 1) && (width_img == IMG_SIZE + 2*NUM_ZERO_PAD - 1))  state   <=  LOAD_INIT_CONV1_WEIGHT;
                LOAD_INIT_CONV1_WEIGHT: if(cnt_init_load_buf2 == C1_OUT_CH*1*C1_K*C1_K-1)                   state   <=  LOAD_INIT_CONV2_WEIGHT;
                LOAD_INIT_CONV2_WEIGHT: if(cnt_init_load_buf2 == C2_OUT_CH*C2_IN_CH*C2_K*C2_K-1)            state   <=  LOAD_INIT_CONV1_BIAS;
                LOAD_INIT_CONV1_BIAS:   if(cnt_init_load_buf2 == C1_OUT_CH-1)                               state   <=  LOAD_INIT_CONV2_BIAS;
                LOAD_INIT_CONV2_BIAS:   if(cnt_init_load_buf2 == C2_OUT_CH-1)                               state   <=  LOAD_INIT_FC1_BIAS;
                LOAD_INIT_FC1_BIAS:     if(cnt_init_load_buf2 == FC1_OUT-1)                                 state   <=  LOAD_INIT_FC2_BIAS;
                LOAD_INIT_FC2_BIAS:     if(cnt_init_load_buf2 == FC2_OUT-1)                                 state   <=  LOAD_INIT_FC3_BIAS;
                LOAD_INIT_FC3_BIAS:     if(cnt_init_load_buf2 == FC3_OUT-1)                                 state   <=  CONV1;
                CONV1:                  if(state_in_conv == SS_END)                                         state   <=  POOL1;
                POOL1:                  if(pool_end)                                                        state   <=  CONV2;
                CONV2:                  if(state_in_conv == SS_END)                                         state   <=  POOL2;
                POOL2:                  if(pool_end)                                                        state   <=  FC1;
                FC1:                    if(state_in_fc == SS_END)                                           state   <=  FC2;
                FC2:                    if(state_in_fc == SS_END)                                           state   <=  FC3;
                FC3:                    if(state_in_fc == SS_END)                                           state   <=  DONE;
            endcase
        end
    end
    
    always @ (posedge clk or posedge reset) begin
        if(reset)   state_in_conv   <=  SS_IDLE;
        else begin
            case(state_in_conv)
                SS_IDLE: if( (state == CONV1) | (state == CONV2) )  state_in_conv   <=  SS_CALL_BIAS;
                SS_CALL_BIAS:                                       state_in_conv   <=  SS_GET_BIAS;
                SS_GET_BIAS:                                        state_in_conv   <=  SS_CONV;
                SS_CONV: begin
                    if( (state == CONV1) & (cnt_cnn_if_start_width == C1_OUT_C-1) & (cnt_cnn_if_start_height == C1_OUT_R-1) & (cnt_cnn_of_ch == C1_OUT_CH-1))       state_in_conv   <=  SS_END;
                    else if( (state == CONV1) & (cnt_cnn_if_start_width == C1_OUT_C-1) & (cnt_cnn_if_start_height == C1_OUT_R-1))                                   state_in_conv   <=  SS_IDLE;
                    else if ( (state == CONV2) & (cnt_cnn_if_start_width == C2_OUT_C-1) & (cnt_cnn_if_start_height == C2_OUT_R-1) & (cnt_cnn_of_ch == C2_OUT_CH-1)) state_in_conv   <=  SS_END;
                    else if ( (state == CONV2) & (cnt_cnn_if_start_width == C2_OUT_C-1) & (cnt_cnn_if_start_height == C2_OUT_R-1))                                  state_in_conv   <=  SS_IDLE;
                end
                SS_END: state_in_conv   <=  SS_IDLE;
            endcase
        end
    end
    
    always @ (posedge clk or posedge reset) begin
        if(reset)   state_in_fc   <=  SS_IDLE;
        else begin
            case(state_in_fc)
                SS_IDLE: if( (state == FC1) | (state == FC2) | (state == FC3) ) state_in_fc   <=  SS_CALL_BIAS;
                SS_CALL_BIAS:                                                   state_in_fc   <=  SS_GET_BIAS;
                SS_GET_BIAS:                                                    state_in_fc   <=  SS_FC;
                SS_FC: begin
                    if((state == FC1)       & (cnt_fc_of == FC1_OUT-1)  & (cnt_fc_if_buf3 == FLAT_SIZE-1))  state_in_fc   <=  SS_END;
                    else if((state == FC1)  & (cnt_fc_if_buf3 == FLAT_SIZE-1))                              state_in_fc   <=  SS_IDLE;
                    else if((state == FC2)  & (cnt_fc_of == FC2_OUT-1)  & (cnt_fc_if_buf3 == FC1_OUT-1))    state_in_fc   <=  SS_END;
                    else if((state == FC2)  & (cnt_fc_if_buf3 == FC1_OUT-1))                                state_in_fc   <=  SS_IDLE;
                    else if((state == FC3)  & (cnt_fc_of == FC3_OUT-1)  & (cnt_fc_if_buf3 == FC2_OUT-1))    state_in_fc   <=  SS_END;
                    else if((state == FC3)  & (cnt_fc_if_buf3 == FC2_OUT-1))                                state_in_fc   <=  SS_IDLE;
                end
                SS_END: state_in_fc   <=  SS_IDLE;
            endcase
        end
    end
    
    height_width_img_gen Uheight_width_img_gen( .reset(reset), .clk(clk), .state(state), .height_img(height_img), .width_img(width_img));
        
    always @ (posedge clk) begin
        if(state == LOAD_IMAGE) begin
            image[width_img][height_img] <= {1'b0, input_data};
        end
    end
    // ========================================
    // mem_ctrl instance
    // ========================================
    mem_ctrl mem_ctrl (
        .reset(reset),
        .clk(clk),
        .state(state),
    
        .cen_conv1_weight_mem(cen_conv1_weight_mem),
        .cen_conv2_weight_mem(cen_conv2_weight_mem),
        .cen_fc1_weight_mem(cen_fc1_weight_mem),
        .cen_fc2_weight_mem(cen_fc2_weight_mem),
        .cen_fc3_weight_mem(cen_fc3_weight_mem),
    
        .addr_conv1_weight_mem(addr_conv1_weight_mem),
        .addr_conv2_weight_mem(addr_conv2_weight_mem),
        .addr_fc1_weight_mem(addr_fc1_weight_mem),
        .addr_fc2_weight_mem(addr_fc2_weight_mem),
        .addr_fc3_weight_mem(addr_fc3_weight_mem),
    
        .cen_conv1_bias_mem(cen_conv1_bias_mem),
        .cen_conv2_bias_mem(cen_conv2_bias_mem),
        .cen_fc1_bias_mem(cen_fc1_bias_mem),
        .cen_fc2_bias_mem(cen_fc2_bias_mem),
        .cen_fc3_bias_mem(cen_fc3_bias_mem),
    
        .addr_conv1_bias_mem(addr_conv1_bias_mem),
        .addr_conv2_bias_mem(addr_conv2_bias_mem),
        .addr_fc1_bias_mem(addr_fc1_bias_mem),
        .addr_fc2_bias_mem(addr_fc2_bias_mem),
        .addr_fc3_bias_mem(addr_fc3_bias_mem),
    
        .cen_conv1_weight_mem_buf1(cen_conv1_weight_mem_buf1),
        .cen_conv2_weight_mem_buf1(cen_conv2_weight_mem_buf1),
        .cen_fc1_weight_mem_buf1(cen_fc1_weight_mem_buf1),
        .cen_fc2_weight_mem_buf1(cen_fc2_weight_mem_buf1),
        .cen_fc3_weight_mem_buf1(cen_fc3_weight_mem_buf1),
    
        .cen_conv1_weight_mem_buf2(cen_conv1_weight_mem_buf2),
        .cen_conv2_weight_mem_buf2(cen_conv2_weight_mem_buf2),
        .cen_fc1_weight_mem_buf2(cen_fc1_weight_mem_buf2),
        .cen_fc2_weight_mem_buf2(cen_fc2_weight_mem_buf2),
        .cen_fc3_weight_mem_buf2(cen_fc3_weight_mem_buf2),
    
        .cen_conv1_bias_mem_buf1(cen_conv1_bias_mem_buf1),
        .cen_conv2_bias_mem_buf1(cen_conv2_bias_mem_buf1),
        .cen_fc1_bias_mem_buf1(cen_fc1_bias_mem_buf1),
        .cen_fc2_bias_mem_buf1(cen_fc2_bias_mem_buf1),
        .cen_fc3_bias_mem_buf1(cen_fc3_bias_mem_buf1),
    
        .cen_conv1_bias_mem_buf2(cen_conv1_bias_mem_buf2),
        .cen_conv2_bias_mem_buf2(cen_conv2_bias_mem_buf2),
        .cen_fc1_bias_mem_buf2(cen_fc1_bias_mem_buf2),
        .cen_fc2_bias_mem_buf2(cen_fc2_bias_mem_buf2),
        .cen_fc3_bias_mem_buf2(cen_fc3_bias_mem_buf2),
    
        .cnt_cnn_height(cnt_cnn_height),
        .cnt_cnn_width(cnt_cnn_width),
        .cnt_cnn_in_ch(cnt_cnn_in_ch),
        .cnt_cnn_out_ch(cnt_cnn_out_ch),
    
        .cnt_init_load(cnt_init_load),
        .cnt_init_load_buf1(cnt_init_load_buf1),
        .cnt_init_load_buf2(cnt_init_load_buf2),        
        
        .state_in_conv              (state_in_conv              ),            
        .cnt_cnn_if_start_width     (cnt_cnn_if_start_width     ),
        .cnt_cnn_if_start_height    (cnt_cnn_if_start_height    ),
        .cnt_cnn_of_ch              (cnt_cnn_of_ch              ),
        
        .state_in_fc                (state_in_fc), 
        .cnt_fc_if                  (cnt_fc_if)
    );
    
    // weight initialize
    always @ (posedge clk) begin
        if((state == LOAD_INIT_CONV1_WEIGHT) & (cen_conv1_weight_mem_buf1))  conv1_weight[cnt_cnn_out_ch][cnt_cnn_in_ch][cnt_cnn_width][cnt_cnn_height] <=  dout_conv1_weight_mem;
        if((state == LOAD_INIT_CONV2_WEIGHT) & (cen_conv2_weight_mem_buf1))  conv2_weight[cnt_cnn_out_ch][cnt_cnn_in_ch][cnt_cnn_width][cnt_cnn_height] <=  dout_conv2_weight_mem;        
    end
    // bias initialize
    always @ (posedge clk) begin
        if((state == LOAD_INIT_CONV1_BIAS) & (cen_conv1_bias_mem_buf1)) conv1_bias[cnt_init_load_buf1] <=  dout_conv1_bias_mem;
        if((state == LOAD_INIT_CONV2_BIAS) & (cen_conv2_bias_mem_buf1)) conv2_bias[cnt_init_load_buf1] <=  dout_conv2_bias_mem;
        if((state == LOAD_INIT_FC1_BIAS) & (cen_fc1_bias_mem_buf1))     fc1_bias[cnt_init_load_buf1] <=  dout_fc1_bias_mem;   
        if((state == LOAD_INIT_FC2_BIAS) & (cen_fc2_bias_mem_buf1))     fc2_bias[cnt_init_load_buf1] <=  dout_fc2_bias_mem;  
        if((state == LOAD_INIT_FC3_BIAS) & (cen_fc3_bias_mem_buf1))     fc3_bias[cnt_init_load_buf1] <=  dout_fc3_bias_mem;       
    end
    
    // conv
    always @ (posedge clk) begin
        if(state == IDLE) begin
            cnt_cnn_if_start_height <=  0;
            cnt_cnn_if_start_width  <=  0;
            curr_cnn_bias           <=  0;
            cnt_cnn_of_ch           <=  0;
        end
        else begin
            case(state_in_conv)
                SS_IDLE: begin
                    cnt_cnn_if_start_height <=  0;
                    cnt_cnn_if_start_width  <=  0;
                    curr_cnn_bias           <=  0;
                end
                SS_GET_BIAS: begin
                    case(state)
                        CONV1:  curr_cnn_bias  <=  dout_conv1_bias_mem;
                        CONV2:  curr_cnn_bias  <=  dout_conv2_bias_mem;
                    endcase
                end
                SS_CONV: begin
                    case(state)
                        CONV1: begin
                            if(cnt_cnn_if_start_height == C1_OUT_R - 1) begin
                                cnt_cnn_if_start_height <=  0;
                                if(cnt_cnn_if_start_width == C1_OUT_C - 1) begin
                                    cnt_cnn_if_start_width  <=  0;
                                    cnt_cnn_of_ch           <=  cnt_cnn_of_ch + 1;
                                end
                                else                                                        cnt_cnn_if_start_width  <=  cnt_cnn_if_start_width + 1;
                            end
                            else cnt_cnn_if_start_height <=  cnt_cnn_if_start_height + 1;
                            conv1_out[cnt_cnn_of_ch][cnt_cnn_if_start_width][cnt_cnn_if_start_height] <= curr_cnn_of;
                        end
                        CONV2: begin
                            if(cnt_cnn_if_start_height == C2_OUT_R - 1) begin
                                cnt_cnn_if_start_height <=  0;
                                if(cnt_cnn_if_start_width == C2_OUT_C - 1) begin
                                    cnt_cnn_if_start_width  <=  0;
                                    cnt_cnn_of_ch           <=  cnt_cnn_of_ch + 1;
                                end
                                else                                                        cnt_cnn_if_start_width  <=  cnt_cnn_if_start_width + 1;
                            end
                            else cnt_cnn_if_start_height <=  cnt_cnn_if_start_height + 1;
                            conv2_out[cnt_cnn_of_ch][cnt_cnn_if_start_width][cnt_cnn_if_start_height] <= curr_cnn_of;
                        end
                    endcase
                end
                SS_END: begin
                    cnt_cnn_if_start_height <=  0;
                    cnt_cnn_if_start_width  <=  0;
                    cnt_cnn_of_ch           <=  0;            
                    curr_cnn_bias           <=  0;    
                end
            endcase
        end
    end
    
    always @ (*) begin
        for(k=0 ; k <6 ; k=k+1) begin
            for(j=0 ; j <5 ; j=j+1) begin
                for(i=0 ; i <5 ; i=i+1) begin
                    curr_cnn_filter[k][j][i]        =   0;
                    curr_cnn_if[k][j][i]            =   0;
                    curr_cnn_pointwise_mul[k][j][i] =   0;                    
                end
            end
        end
        curr_cnn_of_sum =   0;
        curr_cnn_of     =   0;
        case(state)
            CONV1: begin
                for(j=0 ; j<C1_K ; j=j+1) begin // width sweep
                    for(i=0 ; i<C1_K ; i=i+1) begin // height sweep
                        curr_cnn_filter[0][j][i]        =   conv1_weight[cnt_cnn_of_ch][0][j][i]; 
                        curr_cnn_if[0][j][i]            =   image[cnt_cnn_if_start_width + j][cnt_cnn_if_start_height + i];
                        curr_cnn_pointwise_mul[0][j][i] =   curr_cnn_filter[0][j][i] * curr_cnn_if[0][j][i];
                    end    
                end
                curr_cnn_of_sum =   curr_cnn_pointwise_mul[0][0][0] + curr_cnn_pointwise_mul[0][1][0] + curr_cnn_pointwise_mul[0][2][0] + curr_cnn_pointwise_mul[0][3][0] + curr_cnn_pointwise_mul[0][4][0] +
                                    curr_cnn_pointwise_mul[0][0][1] + curr_cnn_pointwise_mul[0][1][1] + curr_cnn_pointwise_mul[0][2][1] + curr_cnn_pointwise_mul[0][3][1] + curr_cnn_pointwise_mul[0][4][1] +
                                    curr_cnn_pointwise_mul[0][0][2] + curr_cnn_pointwise_mul[0][1][2] + curr_cnn_pointwise_mul[0][2][2] + curr_cnn_pointwise_mul[0][3][2] + curr_cnn_pointwise_mul[0][4][2] +
                                    curr_cnn_pointwise_mul[0][0][3] + curr_cnn_pointwise_mul[0][1][3] + curr_cnn_pointwise_mul[0][2][3] + curr_cnn_pointwise_mul[0][3][3] + curr_cnn_pointwise_mul[0][4][3] +
                                    curr_cnn_pointwise_mul[0][0][4] + curr_cnn_pointwise_mul[0][1][4] + curr_cnn_pointwise_mul[0][2][4] + curr_cnn_pointwise_mul[0][3][4] + curr_cnn_pointwise_mul[0][4][4] + 
                                    curr_cnn_bias;
                if(curr_cnn_of_sum < 0)             curr_cnn_of =   0;
                else if(curr_cnn_of_sum > 127)      curr_cnn_of =   127;
                else                                curr_cnn_of = curr_cnn_of_sum[7:0];
            end      
            CONV2: begin
                for(k=0 ; k<P1_OUT_CH ; k=k+1) begin // width sweep
                    for(j=0 ; j<C1_K ; j=j+1) begin // width sweep
                        for(i=0 ; i<C1_K ; i=i+1) begin // height sweep
                            curr_cnn_filter[k][j][i]        =   conv2_weight[cnt_cnn_of_ch][k][j][i]; 
                            curr_cnn_if[k][j][i]            =   pool1_out[k][cnt_cnn_if_start_width + j][cnt_cnn_if_start_height + i];
                            curr_cnn_pointwise_mul[k][j][i] =   curr_cnn_filter[k][j][i] * curr_cnn_if[k][j][i];
                        end    
                    end
                end
                curr_cnn_of_sum =   curr_cnn_pointwise_mul[0][0][0] + curr_cnn_pointwise_mul[0][1][0] + curr_cnn_pointwise_mul[0][2][0] + curr_cnn_pointwise_mul[0][3][0] + curr_cnn_pointwise_mul[0][4][0] +
                                    curr_cnn_pointwise_mul[0][0][1] + curr_cnn_pointwise_mul[0][1][1] + curr_cnn_pointwise_mul[0][2][1] + curr_cnn_pointwise_mul[0][3][1] + curr_cnn_pointwise_mul[0][4][1] +
                                    curr_cnn_pointwise_mul[0][0][2] + curr_cnn_pointwise_mul[0][1][2] + curr_cnn_pointwise_mul[0][2][2] + curr_cnn_pointwise_mul[0][3][2] + curr_cnn_pointwise_mul[0][4][2] +
                                    curr_cnn_pointwise_mul[0][0][3] + curr_cnn_pointwise_mul[0][1][3] + curr_cnn_pointwise_mul[0][2][3] + curr_cnn_pointwise_mul[0][3][3] + curr_cnn_pointwise_mul[0][4][3] +
                                    curr_cnn_pointwise_mul[0][0][4] + curr_cnn_pointwise_mul[0][1][4] + curr_cnn_pointwise_mul[0][2][4] + curr_cnn_pointwise_mul[0][3][4] + curr_cnn_pointwise_mul[0][4][4] +
                                    curr_cnn_pointwise_mul[1][0][0] + curr_cnn_pointwise_mul[1][1][0] + curr_cnn_pointwise_mul[1][2][0] + curr_cnn_pointwise_mul[1][3][0] + curr_cnn_pointwise_mul[1][4][0] +
                                    curr_cnn_pointwise_mul[1][0][1] + curr_cnn_pointwise_mul[1][1][1] + curr_cnn_pointwise_mul[1][2][1] + curr_cnn_pointwise_mul[1][3][1] + curr_cnn_pointwise_mul[1][4][1] +
                                    curr_cnn_pointwise_mul[1][0][2] + curr_cnn_pointwise_mul[1][1][2] + curr_cnn_pointwise_mul[1][2][2] + curr_cnn_pointwise_mul[1][3][2] + curr_cnn_pointwise_mul[1][4][2] +
                                    curr_cnn_pointwise_mul[1][0][3] + curr_cnn_pointwise_mul[1][1][3] + curr_cnn_pointwise_mul[1][2][3] + curr_cnn_pointwise_mul[1][3][3] + curr_cnn_pointwise_mul[1][4][3] +
                                    curr_cnn_pointwise_mul[1][0][4] + curr_cnn_pointwise_mul[1][1][4] + curr_cnn_pointwise_mul[1][2][4] + curr_cnn_pointwise_mul[1][3][4] + curr_cnn_pointwise_mul[1][4][4] +
                                    curr_cnn_pointwise_mul[2][0][0] + curr_cnn_pointwise_mul[2][1][0] + curr_cnn_pointwise_mul[2][2][0] + curr_cnn_pointwise_mul[2][3][0] + curr_cnn_pointwise_mul[2][4][0] +
                                    curr_cnn_pointwise_mul[2][0][1] + curr_cnn_pointwise_mul[2][1][1] + curr_cnn_pointwise_mul[2][2][1] + curr_cnn_pointwise_mul[2][3][1] + curr_cnn_pointwise_mul[2][4][1] +
                                    curr_cnn_pointwise_mul[2][0][2] + curr_cnn_pointwise_mul[2][1][2] + curr_cnn_pointwise_mul[2][2][2] + curr_cnn_pointwise_mul[2][3][2] + curr_cnn_pointwise_mul[2][4][2] +
                                    curr_cnn_pointwise_mul[2][0][3] + curr_cnn_pointwise_mul[2][1][3] + curr_cnn_pointwise_mul[2][2][3] + curr_cnn_pointwise_mul[2][3][3] + curr_cnn_pointwise_mul[2][4][3] +
                                    curr_cnn_pointwise_mul[2][0][4] + curr_cnn_pointwise_mul[2][1][4] + curr_cnn_pointwise_mul[2][2][4] + curr_cnn_pointwise_mul[2][3][4] + curr_cnn_pointwise_mul[2][4][4] +
                                    curr_cnn_pointwise_mul[3][0][0] + curr_cnn_pointwise_mul[3][1][0] + curr_cnn_pointwise_mul[3][2][0] + curr_cnn_pointwise_mul[3][3][0] + curr_cnn_pointwise_mul[3][4][0] +
                                    curr_cnn_pointwise_mul[3][0][1] + curr_cnn_pointwise_mul[3][1][1] + curr_cnn_pointwise_mul[3][2][1] + curr_cnn_pointwise_mul[3][3][1] + curr_cnn_pointwise_mul[3][4][1] +
                                    curr_cnn_pointwise_mul[3][0][2] + curr_cnn_pointwise_mul[3][1][2] + curr_cnn_pointwise_mul[3][2][2] + curr_cnn_pointwise_mul[3][3][2] + curr_cnn_pointwise_mul[3][4][2] +
                                    curr_cnn_pointwise_mul[3][0][3] + curr_cnn_pointwise_mul[3][1][3] + curr_cnn_pointwise_mul[3][2][3] + curr_cnn_pointwise_mul[3][3][3] + curr_cnn_pointwise_mul[3][4][3] +
                                    curr_cnn_pointwise_mul[3][0][4] + curr_cnn_pointwise_mul[3][1][4] + curr_cnn_pointwise_mul[3][2][4] + curr_cnn_pointwise_mul[3][3][4] + curr_cnn_pointwise_mul[3][4][4] +
                                    curr_cnn_pointwise_mul[4][0][0] + curr_cnn_pointwise_mul[4][1][0] + curr_cnn_pointwise_mul[4][2][0] + curr_cnn_pointwise_mul[4][3][0] + curr_cnn_pointwise_mul[4][4][0] +
                                    curr_cnn_pointwise_mul[4][0][1] + curr_cnn_pointwise_mul[4][1][1] + curr_cnn_pointwise_mul[4][2][1] + curr_cnn_pointwise_mul[4][3][1] + curr_cnn_pointwise_mul[4][4][1] +
                                    curr_cnn_pointwise_mul[4][0][2] + curr_cnn_pointwise_mul[4][1][2] + curr_cnn_pointwise_mul[4][2][2] + curr_cnn_pointwise_mul[4][3][2] + curr_cnn_pointwise_mul[4][4][2] +
                                    curr_cnn_pointwise_mul[4][0][3] + curr_cnn_pointwise_mul[4][1][3] + curr_cnn_pointwise_mul[4][2][3] + curr_cnn_pointwise_mul[4][3][3] + curr_cnn_pointwise_mul[4][4][3] +
                                    curr_cnn_pointwise_mul[4][0][4] + curr_cnn_pointwise_mul[4][1][4] + curr_cnn_pointwise_mul[4][2][4] + curr_cnn_pointwise_mul[4][3][4] + curr_cnn_pointwise_mul[4][4][4] +
                                    curr_cnn_pointwise_mul[5][0][0] + curr_cnn_pointwise_mul[5][1][0] + curr_cnn_pointwise_mul[5][2][0] + curr_cnn_pointwise_mul[5][3][0] + curr_cnn_pointwise_mul[5][4][0] +
                                    curr_cnn_pointwise_mul[5][0][1] + curr_cnn_pointwise_mul[5][1][1] + curr_cnn_pointwise_mul[5][2][1] + curr_cnn_pointwise_mul[5][3][1] + curr_cnn_pointwise_mul[5][4][1] +
                                    curr_cnn_pointwise_mul[5][0][2] + curr_cnn_pointwise_mul[5][1][2] + curr_cnn_pointwise_mul[5][2][2] + curr_cnn_pointwise_mul[5][3][2] + curr_cnn_pointwise_mul[5][4][2] +
                                    curr_cnn_pointwise_mul[5][0][3] + curr_cnn_pointwise_mul[5][1][3] + curr_cnn_pointwise_mul[5][2][3] + curr_cnn_pointwise_mul[5][3][3] + curr_cnn_pointwise_mul[5][4][3] +
                                    curr_cnn_pointwise_mul[5][0][4] + curr_cnn_pointwise_mul[5][1][4] + curr_cnn_pointwise_mul[5][2][4] + curr_cnn_pointwise_mul[5][3][4] + curr_cnn_pointwise_mul[5][4][4];
                                    
                curr_cnn_of_sum_rounded_by_16 = (curr_cnn_of_sum + 8) >>> 4;
                curr_cnn_of_sum_rounded_by_16_plus_bias = curr_cnn_of_sum_rounded_by_16 + curr_cnn_bias; 
                if(curr_cnn_of_sum_rounded_by_16_plus_bias < 0)         curr_cnn_of =   0;
                else if(curr_cnn_of_sum_rounded_by_16_plus_bias > 127)  curr_cnn_of =   127;
                else                                                    curr_cnn_of =   curr_cnn_of_sum_rounded_by_16_plus_bias[7:0];
            end       
        endcase
    end
    
    // Pool 
    always @ (posedge clk) begin
        case(state)
            IDLE: begin
                cnt_pool_if_start_height    <=  0;
                cnt_pool_if_start_width     <=  0;
                cnt_pool_if_ch              <=  0;
            end
            POOL1: begin
                if(cnt_pool_if_start_height == P1_OUT_R - 1) begin
                    cnt_pool_if_start_height <=  0;
                    if(cnt_pool_if_start_width == P1_OUT_C - 1) begin
                        cnt_pool_if_start_width  <=  0;
                        if(cnt_pool_if_ch == C1_OUT_CH - 1) cnt_pool_if_ch  <=  0;
                        else                                cnt_pool_if_ch  <=  cnt_pool_if_ch + 1;
                    end
                    else cnt_pool_if_start_width  <=  cnt_pool_if_start_width + 1;
                end
                else cnt_pool_if_start_height <=  cnt_pool_if_start_height + 1;
                pool1_out[cnt_pool_if_ch][cnt_pool_if_start_width][cnt_pool_if_start_height]  <= curr_pool_of;                
            end
            POOL2: begin
                if(cnt_pool_if_start_height == P2_OUT_R - 1) begin
                    cnt_pool_if_start_height <=  0;
                    if(cnt_pool_if_start_width == P2_OUT_C - 1) begin
                        cnt_pool_if_start_width  <=  0;
                        if(cnt_pool_if_ch == C2_OUT_CH - 1) cnt_pool_if_ch  <=  0;
                        else                                cnt_pool_if_ch  <=  cnt_pool_if_ch + 1;
                    end
                    else cnt_pool_if_start_width  <=  cnt_pool_if_start_width + 1;
                end
                else cnt_pool_if_start_height <=  cnt_pool_if_start_height + 1;
                pool2_out[cnt_pool_if_ch][cnt_pool_if_start_width][cnt_pool_if_start_height]  <= curr_pool_of;    
            end
        endcase
    end
    
    always @ (*) begin
        pool_end    =   0;
        for(j=0 ; j<2 ; j=j+1) begin // width sweep
            for(i=0 ; i<2 ; i=i+1) begin // height sweep
                curr_pool_if[j][i] = 0;
            end
        end    
        curr_pool_of_sum = 0;
        curr_pool_of = 0;
        case(state)
            POOL1: begin 
                pool_end = (cnt_pool_if_start_height == P1_OUT_R - 1) & (cnt_pool_if_start_width == P1_OUT_C - 1) & (cnt_pool_if_ch == C1_OUT_CH - 1);
                for(j=0 ; j<2 ; j=j+1) begin // width sweep
                    for(i=0 ; i<2 ; i=i+1) begin // height sweep
                        curr_pool_if[j][i] = conv1_out[cnt_pool_if_ch][2*cnt_pool_if_start_width + j][2*cnt_pool_if_start_height + i];
                    end
                end    
                curr_pool_of_sum        =   curr_pool_if[0][0] + curr_pool_if[1][0] + 
                                            curr_pool_if[0][1] + curr_pool_if[1][1];
//                curr_pool_of_sum_round  =   (curr_pool_of_sum >= 0) ? (curr_pool_of_sum + 2) : (curr_pool_of_sum - 2);
                curr_pool_of_sum_round  =   (curr_pool_of_sum + 2);
                curr_pool_of            =   curr_pool_of_sum_round >>> 2;
            end     
            POOL2: begin 
                pool_end = (cnt_pool_if_start_height == P2_OUT_R - 1) & (cnt_pool_if_start_width == P2_OUT_C - 1) & (cnt_pool_if_ch == C2_OUT_CH - 1);
                for(j=0 ; j<2 ; j=j+1) begin // width sweep
                    for(i=0 ; i<2 ; i=i+1) begin // height sweep
                        curr_pool_if[j][i] = conv2_out[cnt_pool_if_ch][2*cnt_pool_if_start_width + j][2*cnt_pool_if_start_height + i];
                    end
                end    
                curr_pool_of_sum        =   curr_pool_if[0][0] + curr_pool_if[1][0] + 
                                            curr_pool_if[0][1] + curr_pool_if[1][1];
//                curr_pool_of_sum_round  =   (curr_pool_of_sum >= 0) ? (curr_pool_of_sum + 2) : (curr_pool_of_sum - 2);
                curr_pool_of_sum_round  =   (curr_pool_of_sum + 2);
                curr_pool_of            =   curr_pool_of_sum_round >>> 2;
            end         
        endcase
    end
    
    // Flatten
    always @ (*) begin
        for(k=0 ; k<P2_OUT_CH ; k=k+1) begin
            for(j=0 ; j<P2_OUT_C ; j=j+1) begin
                for(i=0 ; i<P2_OUT_R ; i=i+1) begin
                    pool2_flat[k + P2_OUT_CH * j + P2_OUT_CH * P2_OUT_C * i] = pool2_out[k][j][i];
                end
            end
        end
    end
    
    // FC
    always @ (posedge clk) begin
        case(state_in_fc)
            SS_IDLE:curr_fc_bias    <=  0;
            SS_GET_BIAS:begin
                case(state)
                    FC1: curr_fc_bias    <=  dout_fc1_bias_mem;
                    FC2: curr_fc_bias    <=  dout_fc2_bias_mem;
                    FC3: curr_fc_bias    <=  dout_fc3_bias_mem;
                endcase
            end
            SS_END: curr_fc_bias    <=  0;            
        endcase
        
        if (state == IDLE)                                      curr_fc_weight   <=  0;
        else if ((state == FC1) & (cen_fc1_weight_mem_buf1))    curr_fc_weight   <=  dout_fc1_weight_mem;
        else if ((state == FC2) & (cen_fc2_weight_mem_buf1))    curr_fc_weight   <=  dout_fc2_weight_mem;
        else if ((state == FC3) & (cen_fc3_weight_mem_buf1))    curr_fc_weight   <=  dout_fc3_weight_mem;
    end
    
    always @ (*) begin
        curr_fc_if = 0;
        case(state)
            FC1: curr_fc_if  =   pool2_flat[cnt_fc_if_buf2];
            FC2: curr_fc_if  =   fc1_out[cnt_fc_if_buf2];
            FC3: curr_fc_if  =   fc2_out[cnt_fc_if_buf2];
        endcase                
    end
    
    always @ (posedge clk) begin
        if(state_in_fc == SS_IDLE)                      cnt_fc_if   <=  0;
        else if((state == FC1) & (cen_fc1_weight_mem))  cnt_fc_if   <=  cnt_fc_if + 1;
        else if((state == FC2) & (cen_fc2_weight_mem))  cnt_fc_if   <=  cnt_fc_if + 1;
        else if((state == FC3) & (cen_fc3_weight_mem))  cnt_fc_if   <=  cnt_fc_if + 1;
        
        cnt_fc_if_buf1  <=  cnt_fc_if;
        cnt_fc_if_buf2  <=  cnt_fc_if_buf1;
        cnt_fc_if_buf3  <=  cnt_fc_if_buf2;
    end
    
    always @ (posedge clk) begin
        if(reset)                                                   cnt_fc_of   <=  0;
        else if(state == IDLE)                                      cnt_fc_of   <=  0;
        else if(state_in_fc == SS_END)                              cnt_fc_of   <=  0;
        else if( (state == FC1) & (cnt_fc_if_buf3 == FLAT_SIZE-1) ) cnt_fc_of   <=  cnt_fc_of + 1;
        else if( (state == FC2) & (cnt_fc_if_buf3 == FC1_OUT-1) )   cnt_fc_of   <=  cnt_fc_of + 1;
        else if( (state == FC3) & (cnt_fc_if_buf3 == FC2_OUT-1) )   cnt_fc_of   <=  cnt_fc_of + 1;
    end
    
    always @ (posedge clk) begin
        case(state)
            FC1: begin
                if(state_in_fc == SS_IDLE)          curr_fc_of_sum  <=  0;
                else if(cen_fc1_weight_mem_buf2)    curr_fc_of_sum  <=  curr_fc_of_sum + curr_fc_weight * curr_fc_if;
            end
            FC2: begin
                if(state_in_fc == SS_IDLE)          curr_fc_of_sum  <=  0;
                else if(cen_fc2_weight_mem_buf2)    curr_fc_of_sum  <=  curr_fc_of_sum + curr_fc_weight * curr_fc_if;
            end
            FC3: begin
                if(state_in_fc == SS_IDLE)          curr_fc_of_sum  <=  0;
                else if(cen_fc3_weight_mem_buf2)    curr_fc_of_sum  <=  curr_fc_of_sum + curr_fc_weight * curr_fc_if;
            end
        endcase
    end
    
    always @ (*) begin
        case(state)
            FC1: curr_fc_of_sum_round4 = (curr_fc_of_sum + 8) >>> 4;
            FC2: curr_fc_of_sum_round4 = (curr_fc_of_sum + 8) >>> 4;
            FC3: curr_fc_of_sum_round4 = (curr_fc_of_sum + 8) >>> 4;
        endcase
        curr_fc_of_sum_round4_plus_bias = curr_fc_of_sum_round4 + curr_fc_bias;
    end
    
    always @ (*) begin
        curr_fc_of = 0;
        case(state)
            FC1: begin
                if((state_in_fc == SS_FC) & (cnt_fc_if_buf3 == FLAT_SIZE-1)) begin
                    if(curr_fc_of_sum_round4_plus_bias < 0 )        curr_fc_of  =   0;
                    else if(curr_fc_of_sum_round4_plus_bias > 127)  curr_fc_of  =   127;
                    else                                            curr_fc_of  =   curr_fc_of_sum_round4_plus_bias;
                end  
            end
            FC2: begin
                if((state_in_fc == SS_FC) & (cnt_fc_if_buf3 == FC1_OUT-1)) begin
                    if(curr_fc_of_sum_round4_plus_bias < 0 )        curr_fc_of  =   0;
                    else if(curr_fc_of_sum_round4_plus_bias > 127)  curr_fc_of  =   127;
                    else                                            curr_fc_of  =   curr_fc_of_sum_round4_plus_bias;
                end  
            end
            FC3: begin
                if((state_in_fc == SS_FC) & (cnt_fc_if_buf3 == FC2_OUT-1)) begin
                    curr_fc_of  =   curr_fc_of_sum_round4_plus_bias;
                end  
            end
        endcase
    end
    
    
    always @ (posedge clk) begin
        case(state)
            FC1: if(cnt_fc_if_buf3 == FLAT_SIZE-1)  fc1_out[cnt_fc_of]  <=  curr_fc_of;
            FC2: if(cnt_fc_if_buf3 == FC1_OUT-1)    fc2_out[cnt_fc_of]  <=  curr_fc_of;
            FC3: if(cnt_fc_if_buf3 == FC2_OUT-1)    fc3_out[cnt_fc_of]  <=  curr_fc_of;
        endcase
    end
    
    
    
    always @ (*) begin
        for(i=0; i<NUM_CLASSES; i=i+1) begin
            max_val[i]         =   0;
            max_index[i]       =   0;
        end
        output_class    =   0;
        done            =   0;
        if(state == DONE) begin            
            max_val[0]      =   fc3_out[0];
            max_index[0]    =   0;
            for(i=0 ; i<NUM_CLASSES-1; i=i+1) begin
                if(max_val[i] < fc3_out[i+1]) begin
                    max_val[i+1]  =   fc3_out[i+1];
                    max_index[i+1]=   i+1;
                end
                else begin
                    max_val[i+1]  =   max_val[i];
                    max_index[i+1]=   max_index[i];
                end            
            end
            output_class = max_index[NUM_CLASSES-1];
            done = 1;
        end
    end
endmodule