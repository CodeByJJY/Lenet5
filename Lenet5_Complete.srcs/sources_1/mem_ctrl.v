module mem_ctrl( 
    input   wire            reset,
    input   wire            clk,
    input   wire    [4:0]   state,
   
    output reg         cen_conv1_weight_mem,   cen_conv2_weight_mem,   cen_fc1_weight_mem,    cen_fc2_weight_mem,    cen_fc3_weight_mem,

    output reg [7:0] addr_conv1_weight_mem,
    output reg [11:0] addr_conv2_weight_mem,
    output reg [15:0] addr_fc1_weight_mem,
    output reg [13:0] addr_fc2_weight_mem,
    output reg [9:0] addr_fc3_weight_mem,
    
    
    output reg         cen_conv1_bias_mem,     cen_conv2_bias_mem,     cen_fc1_bias_mem,      cen_fc2_bias_mem,      cen_fc3_bias_mem,

    output reg [2:0] addr_conv1_bias_mem,
    output reg [3:0] addr_conv2_bias_mem,
    output reg [6:0] addr_fc1_bias_mem,
    output reg [6:0] addr_fc2_bias_mem,
    output reg [3:0] addr_fc3_bias_mem,

    output reg cen_conv1_weight_mem_buf1,  cen_conv2_weight_mem_buf1,  cen_fc1_weight_mem_buf1,   cen_fc2_weight_mem_buf1,   cen_fc3_weight_mem_buf1,
    output reg cen_conv1_weight_mem_buf2,  cen_conv2_weight_mem_buf2,  cen_fc1_weight_mem_buf2,   cen_fc2_weight_mem_buf2,   cen_fc3_weight_mem_buf2,
        
    output reg cen_conv1_bias_mem_buf1,  cen_conv2_bias_mem_buf1,  cen_fc1_bias_mem_buf1,   cen_fc2_bias_mem_buf1,   cen_fc3_bias_mem_buf1,
    output reg cen_conv1_bias_mem_buf2,  cen_conv2_bias_mem_buf2,  cen_fc1_bias_mem_buf2,   cen_fc2_bias_mem_buf2,   cen_fc3_bias_mem_buf2,
    
    output reg[2:0] cnt_cnn_height,
    output reg[2:0] cnt_cnn_width,
    output reg[3:0] cnt_cnn_in_ch,
    output reg[3:0] cnt_cnn_out_ch,
    
    output reg [31:0] cnt_init_load,
    output reg [31:0] cnt_init_load_buf1,
    output reg [31:0] cnt_init_load_buf2,
    
    input wire  [2:0] state_in_conv,
        
    input wire  [4:0] cnt_cnn_if_start_width,
    input wire  [4:0] cnt_cnn_if_start_height,
    input wire  [3:0] cnt_cnn_of_ch,
    
    input wire  [2:0] state_in_fc,    
    input wire  [8:0] cnt_fc_if
);


//    reg cen_conv1_weight_mem_buf1,  cen_conv2_weight_mem_buf1,  cen_fc1_weight_mem_buf1,   cen_fc2_weight_mem_buf1,   cen_fc3_weight_mem_buf1;
//    reg cen_conv1_weight_mem_buf2,  cen_conv2_weight_mem_buf2,  cen_fc1_weight_mem_buf2,   cen_fc2_weight_mem_buf2,   cen_fc3_weight_mem_buf2;
    
//    reg cen_conv1_bias_mem_buf1,  cen_conv2_bias_mem_buf1,  cen_fc1_bias_mem_buf1,   cen_fc2_bias_mem_buf1,   cen_fc3_bias_mem_buf1;
//    reg cen_conv1_bias_mem_buf2,  cen_conv2_bias_mem_buf2,  cen_fc1_bias_mem_buf2,   cen_fc2_bias_mem_buf2,   cen_fc3_bias_mem_buf2;

//    reg[2:0] cnt_cnn_height;
//    reg[2:0] cnt_cnn_width;
//    reg[3:0] cnt_cnn_in_ch;
//    reg[3:0] cnt_cnn_out_ch;

//    reg [31:0] cnt_init_load;
//    reg [31:0] cnt_init_load_buf1;
    
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
                
                 
    always @ (*) begin
        cen_conv1_weight_mem = 0;
        if( (state == LOAD_INIT_CONV1_WEIGHT) & (cnt_init_load < C1_OUT_CH*1*C1_K*C1_K))    cen_conv1_weight_mem    =   1;
    end    
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cen_conv1_weight_mem_buf1   <=  0;
            cen_conv1_weight_mem_buf2   <=  0;
            addr_conv1_weight_mem       <=  0;            
        end
        else begin
            cen_conv1_weight_mem_buf1   <=  cen_conv1_weight_mem;
            cen_conv1_weight_mem_buf2   <=  cen_conv1_weight_mem_buf2;
            if((state == LOAD_INIT_CONV1_WEIGHT) & (cnt_init_load_buf2 == C1_OUT_CH*1*C1_K*C1_K - 1))   addr_conv1_weight_mem   <=  0;
            else if(state == LOAD_INIT_CONV1_WEIGHT)                                                    addr_conv1_weight_mem   <=  addr_conv1_weight_mem + 1;            
        end
    end
    
    always @ (*) begin
        cen_conv2_weight_mem = 0;
        if( (state == LOAD_INIT_CONV2_WEIGHT) & (cnt_init_load < C2_OUT_CH*C2_IN_CH*C2_K*C2_K))    cen_conv2_weight_mem    =   1;        
    end
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cen_conv2_weight_mem_buf1   <=  0;
            cen_conv2_weight_mem_buf2   <=  0;
            addr_conv2_weight_mem       <=  0;            
        end
        else begin
            cen_conv2_weight_mem_buf1   <=  cen_conv2_weight_mem;
            cen_conv2_weight_mem_buf2   <=  cen_conv2_weight_mem_buf2;
            if((state == LOAD_INIT_CONV2_WEIGHT) & (cnt_init_load_buf2 == C2_OUT_CH*C2_IN_CH*C2_K*C2_K - 1))    addr_conv2_weight_mem   <=  0;
            else if(state == LOAD_INIT_CONV2_WEIGHT)                                                            addr_conv2_weight_mem   <=  addr_conv2_weight_mem + 1;
        end
    end
    
    always @ (*) begin
        cen_fc1_weight_mem  =   0;
        if((state == FC1) & (state_in_fc != SS_IDLE) & (cnt_fc_if < FLAT_SIZE))  cen_fc1_weight_mem  =   1;
    end
    always @ (posedge clk or posedge reset) begin   
        if(reset) begin
            cen_fc1_weight_mem_buf1 <=  0;
            cen_fc1_weight_mem_buf2 <=  0;
        end
        else begin
            cen_fc1_weight_mem_buf1 <=  cen_fc1_weight_mem;
            cen_fc1_weight_mem_buf2 <=  cen_fc1_weight_mem_buf1;
        end
    end    
    always @ (posedge clk or posedge reset) begin
        if(reset) addr_fc1_weight_mem <=  0;
        else begin
            if((state == FC1) & (cen_fc1_weight_mem))           addr_fc1_weight_mem <=  addr_fc1_weight_mem + 1;
            else if((state == FC1) & (state_in_fc == SS_END))   addr_fc1_weight_mem <=  0;
        end        
    end
    
    always @ (*) begin
        cen_fc2_weight_mem  =   0;
        if((state == FC2) & (state_in_fc != SS_IDLE) & (cnt_fc_if < FC1_OUT))  cen_fc2_weight_mem  =   1;
    end    
    always @ (posedge clk or posedge reset) begin   
        if(reset) begin
            cen_fc2_weight_mem_buf1 <=  0;
            cen_fc2_weight_mem_buf2 <=  0;
        end
        else begin
            cen_fc2_weight_mem_buf1 <=  cen_fc2_weight_mem;
            cen_fc2_weight_mem_buf2 <=  cen_fc2_weight_mem_buf1;
        end
    end  
    always @ (posedge clk or posedge reset) begin
        if(reset) addr_fc2_weight_mem <=  0;
        else begin
            if((state == FC2) & (cen_fc2_weight_mem))           addr_fc2_weight_mem <=  addr_fc2_weight_mem + 1;
            else if((state == FC2) & (state_in_fc == SS_END))   addr_fc2_weight_mem <=  0;
        end        
    end
    
    always @ (*) begin
        cen_fc3_weight_mem  =   0;
        if((state == FC3) & (state_in_fc != SS_IDLE) & (cnt_fc_if < FC2_OUT))  cen_fc3_weight_mem  =   1;
    end    
    always @ (posedge clk or posedge reset) begin   
        if(reset) begin
            cen_fc3_weight_mem_buf1 <=  0;
            cen_fc3_weight_mem_buf2 <=  0;
        end
        else begin
            cen_fc3_weight_mem_buf1 <=  cen_fc3_weight_mem;
            cen_fc3_weight_mem_buf2 <=  cen_fc3_weight_mem_buf1;
        end
    end  
    always @ (posedge clk or posedge reset) begin
        if(reset) addr_fc3_weight_mem <=  0;
        else begin
            if((state == FC3) & (cen_fc3_weight_mem))           addr_fc3_weight_mem <=  addr_fc3_weight_mem + 1;
            else if((state == FC3) & (state_in_fc == SS_END))   addr_fc3_weight_mem <=  0;
        end        
    end
    
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cnt_cnn_height  <=  0;
            cnt_cnn_width   <=  0;
            cnt_cnn_in_ch   <=  0;
            cnt_cnn_out_ch  <=  0;
        end
        else begin
            case(state)
                LOAD_INIT_CONV1_WEIGHT: begin
                    if(cen_conv1_weight_mem_buf1) begin
                        if(cnt_cnn_height == C1_K-1)    cnt_cnn_height  <=  0;
                        else                            cnt_cnn_height  <=  cnt_cnn_height + 1;
                        if((cnt_cnn_height == C1_K-1) & (cnt_cnn_width == C1_K-1))  cnt_cnn_width   <=  0;
                        else if(cnt_cnn_height == C1_K-1)                           cnt_cnn_width   <=  cnt_cnn_width + 1;
                        cnt_cnn_in_ch   <=  0;  //because C1_IN_CH == 1
                        if((cnt_cnn_height == C1_K-1) & (cnt_cnn_width == C1_K-1) & (cnt_cnn_out_ch == C1_OUT_CH-1))    cnt_cnn_out_ch  <=  0;
                        else if((cnt_cnn_height == C1_K-1) & (cnt_cnn_width == C1_K-1))                                 cnt_cnn_out_ch  <=  cnt_cnn_out_ch + 1;
                    end
                end
                LOAD_INIT_CONV2_WEIGHT: begin
                    if(cen_conv2_weight_mem_buf1) begin
                        if(cnt_cnn_height == C2_K-1)    cnt_cnn_height  <=  0;
                        else                            cnt_cnn_height  <=  cnt_cnn_height + 1;
                        if((cnt_cnn_height == C2_K-1) & (cnt_cnn_width == C2_K-1))  cnt_cnn_width   <=  0;
                        else if(cnt_cnn_height == C2_K-1)                           cnt_cnn_width   <=  cnt_cnn_width + 1;
                        if((cnt_cnn_height == C2_K-1) & (cnt_cnn_width == C2_K-1) & (cnt_cnn_in_ch == C2_IN_CH-1))  cnt_cnn_in_ch  <=  0;
                        else if((cnt_cnn_height == C2_K-1) & (cnt_cnn_width == C2_K-1))                             cnt_cnn_in_ch  <=  cnt_cnn_in_ch + 1;
                        if((cnt_cnn_height == C2_K-1) & (cnt_cnn_width == C2_K-1) & (cnt_cnn_in_ch == C2_IN_CH-1) & (cnt_cnn_out_ch == C2_OUT_CH-1))    cnt_cnn_out_ch  <=  0;
                        else if((cnt_cnn_height == C2_K-1) & (cnt_cnn_width == C2_K-1) & (cnt_cnn_in_ch == C2_IN_CH-1))                                 cnt_cnn_out_ch  <=  cnt_cnn_out_ch + 1;
                    end
                end
            endcase
        end
    end
    
    always @ (*) begin
        cen_conv1_bias_mem = 0;
        if((state == LOAD_INIT_CONV1_BIAS) & (cnt_init_load < C1_OUT_CH))   cen_conv1_bias_mem  =   1;        
        else if( (state == CONV1) & (state_in_conv == SS_CALL_BIAS) )       cen_conv1_bias_mem  =   1;
    end
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cen_conv1_bias_mem_buf1 <=  0;
            cen_conv1_bias_mem_buf2 <=  0;  
            addr_conv1_bias_mem     <=  0;            
        end
        else begin
            cen_conv1_bias_mem_buf1 <=  cen_conv1_bias_mem;
            cen_conv1_bias_mem_buf2 <=  cen_conv1_bias_mem_buf1;        
            if((state == LOAD_INIT_CONV1_BIAS) & (cnt_init_load_buf2 == C1_OUT_CH - 1)) addr_conv1_bias_mem <=  0;
            else if(state == LOAD_INIT_CONV1_BIAS)                                      addr_conv1_bias_mem <=  addr_conv1_bias_mem + 1;
            else if((state == CONV1) & (cen_conv1_bias_mem))                            addr_conv1_bias_mem <=  addr_conv1_bias_mem + 1;
            else if((state == CONV1) & (state_in_conv == SS_END))                       addr_conv1_bias_mem <=  0;
        end
    end
    always @ (*) begin
        cen_conv2_bias_mem = 0;
        if((state == LOAD_INIT_CONV2_BIAS) & (cnt_init_load < C2_OUT_CH))   cen_conv2_bias_mem  =   1;    
        else if( (state == CONV2) & (state_in_conv == SS_CALL_BIAS) )       cen_conv2_bias_mem  =   1;    
    end
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cen_conv2_bias_mem_buf1 <=  0;
            cen_conv2_bias_mem_buf2 <=  0;   
            addr_conv2_bias_mem     <=  0;             
        end
        else begin
            cen_conv2_bias_mem_buf1 <=  cen_conv2_bias_mem;
            cen_conv2_bias_mem_buf2 <=  cen_conv2_bias_mem_buf1;    
            if((state == LOAD_INIT_CONV2_BIAS) & (cnt_init_load_buf2 == C2_OUT_CH - 1)) addr_conv2_bias_mem   <=  0;
            else if(state == LOAD_INIT_CONV2_BIAS)                                      addr_conv2_bias_mem   <=  addr_conv2_bias_mem + 1;
            else if((state == CONV2) & (cen_conv2_bias_mem))                            addr_conv2_bias_mem <=  addr_conv2_bias_mem + 1;
            else if((state == CONV2) & (state_in_conv == SS_END))                       addr_conv2_bias_mem <=  0;              
        end
    end
    always @ (*) begin
        cen_fc1_bias_mem = 0;
        if((state == LOAD_INIT_FC1_BIAS) & (cnt_init_load < FC1_OUT))   cen_fc1_bias_mem    =   1;        
        else if((state == FC1) & (state_in_fc == SS_CALL_BIAS))         cen_fc1_bias_mem    =   1;
    end
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cen_fc1_bias_mem_buf1 <=  0;
            cen_fc1_bias_mem_buf2 <=  0;     
            addr_fc1_bias_mem     <=  0;         
        end
        else begin
            cen_fc1_bias_mem_buf1 <=  cen_fc1_bias_mem;
            cen_fc1_bias_mem_buf2 <=  cen_fc1_bias_mem_buf1;   
            if((state == LOAD_INIT_FC1_BIAS) & (cnt_init_load_buf2 == FC1_OUT - 1)) addr_fc1_bias_mem   <=  0;
            else if(state == LOAD_INIT_FC1_BIAS)                                    addr_fc1_bias_mem   <=  addr_fc1_bias_mem + 1;
            else if((state == FC1) & (cen_fc1_bias_mem))                            addr_fc1_bias_mem   <=  addr_fc1_bias_mem + 1;
            else if((state == FC1) & (state_in_fc == SS_END))                       addr_fc1_bias_mem   <=  0;
        end
    end
    always @ (*) begin
        cen_fc2_bias_mem = 0;
        if((state == LOAD_INIT_FC2_BIAS) & (cnt_init_load < FC2_OUT))   cen_fc2_bias_mem    =   1;    
        else if((state == FC2) & (state_in_fc == SS_CALL_BIAS))         cen_fc2_bias_mem    =   1;    
    end
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cen_fc2_bias_mem_buf1 <=  0;
            cen_fc2_bias_mem_buf2 <=  0;    
            addr_fc2_bias_mem     <=  0;           
        end
        else begin
            cen_fc2_bias_mem_buf1 <=  cen_fc2_bias_mem;
            cen_fc2_bias_mem_buf2 <=  cen_fc2_bias_mem_buf1;   
            if((state == LOAD_INIT_FC2_BIAS) & (cnt_init_load_buf2 == FC2_OUT - 1)) addr_fc2_bias_mem   <=  0;
            else if(state == LOAD_INIT_FC2_BIAS)                                    addr_fc2_bias_mem   <=  addr_fc2_bias_mem + 1;  
            else if((state == FC2) & (cen_fc2_bias_mem))                            addr_fc2_bias_mem   <=  addr_fc2_bias_mem + 1;
            else if((state == FC2) & (state_in_fc == SS_END))                       addr_fc2_bias_mem   <=  0;            
        end
    end
    always @ (*) begin
        cen_fc3_bias_mem = 0;
        if((state == LOAD_INIT_FC3_BIAS) & (cnt_init_load < FC3_OUT))   cen_fc3_bias_mem    =   1;      
        else if((state == FC3) & (state_in_fc == SS_CALL_BIAS))         cen_fc3_bias_mem    =   1;      
    end
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cen_fc3_bias_mem_buf1 <=  0;
            cen_fc3_bias_mem_buf2 <=  0;  
            addr_fc3_bias_mem     <=  0;                
        end
        else begin
            cen_fc3_bias_mem_buf1 <=  cen_fc3_bias_mem;
            cen_fc3_bias_mem_buf2 <=  cen_fc3_bias_mem_buf1;     
            if((state == LOAD_INIT_FC3_BIAS) & (cnt_init_load_buf2 == FC3_OUT - 1)) addr_fc3_bias_mem   <=  0;
            else if(state == LOAD_INIT_FC3_BIAS)                                    addr_fc3_bias_mem   <=  addr_fc3_bias_mem + 1;  
            else if((state == FC3) & (cen_fc3_bias_mem))                            addr_fc3_bias_mem   <=  addr_fc3_bias_mem + 1;
            else if((state == FC3) & (state_in_fc == SS_END))                       addr_fc3_bias_mem   <=  0;                     
        end
    end    
    
    always @ (posedge clk or posedge reset) begin
        if(reset) begin
            cnt_init_load   <=  0;
            cnt_init_load_buf1   <=  0;
            cnt_init_load_buf2   <=  0;
        end
        else begin
            cnt_init_load_buf1   <=  cnt_init_load;
            cnt_init_load_buf2   <=  cnt_init_load_buf1;
            case(state)
                LOAD_INIT_CONV1_WEIGHT: begin   
                    if(cnt_init_load_buf2 == C1_OUT_CH*1*C1_K*C1_K - 1) cnt_init_load   <=  0;
                    else                                                cnt_init_load   <=  cnt_init_load + 1;         
                end
                LOAD_INIT_CONV2_WEIGHT: begin
                    if(cnt_init_load_buf2 == C2_OUT_CH*C2_IN_CH*C2_K*C2_K - 1)  cnt_init_load   <=  0;
                    else                                                        cnt_init_load   <=  cnt_init_load + 1; 
                end
                LOAD_INIT_CONV1_BIAS  : begin
                    if(cnt_init_load_buf2 == C1_OUT_CH - 1)             cnt_init_load   <=  0;
                    else                                                cnt_init_load   <=  cnt_init_load + 1;     
                end
                LOAD_INIT_CONV2_BIAS  : begin
                    if(cnt_init_load_buf2 == C2_OUT_CH - 1)             cnt_init_load   <=  0;
                    else                                                cnt_init_load   <=  cnt_init_load + 1;     
                end
                LOAD_INIT_FC1_BIAS    : begin
                    if(cnt_init_load_buf2 == FC1_OUT - 1)               cnt_init_load   <=  0;
                    else                                                cnt_init_load   <=  cnt_init_load + 1; 
                end
                LOAD_INIT_FC2_BIAS    : begin
                    if(cnt_init_load_buf2 == FC2_OUT - 1)               cnt_init_load   <=  0;
                    else                                                cnt_init_load   <=  cnt_init_load + 1; 
                end
                LOAD_INIT_FC3_BIAS : begin
                    if(cnt_init_load_buf2 == FC3_OUT - 1)               cnt_init_load   <=  0;
                    else                                                cnt_init_load   <=  cnt_init_load + 1; 
                end
            endcase
        end
    end

endmodule
