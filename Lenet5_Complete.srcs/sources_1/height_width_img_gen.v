module height_width_img_gen(
    input   wire        reset,
    input   wire        clk,
    input   wire [4:0]  state,
    output  reg [4:0]   width_img,
    output  reg [4:0]   height_img
);


    localparam IMG_SIZE     = 28;
    localparam NUM_ZERO_PAD = 2;

    // State Machine States
    localparam  IDLE                    = 0, 
                LOAD_IMAGE              = 1;


    always @(posedge clk or posedge reset) begin 
        if(reset) begin
            height_img <=  0;
            width_img <=  0;
        end
        else begin
            case(state)
                IDLE: begin
                    height_img <=  0;
                    width_img <=  0;                    
                end
                LOAD_IMAGE: begin
                    if(width_img == IMG_SIZE + 2*NUM_ZERO_PAD - 1) begin
                        width_img <=  0;
                        if(height_img == IMG_SIZE + 2*NUM_ZERO_PAD - 1) height_img <=  0;
                        else                                            height_img <=  height_img + 1;
                    end
                    else width_img <=  width_img + 1;
                end
            endcase
        end
    end

endmodule
