1.对于python backend来说：
    默认是CPU
    instance_group [{ kind: KIND_CPU }]可有可无
2.如果想在python backend上指定GPU要用：
    parameters:{
    key:"FORCE_CPU_ONLY_INPUT_TENSORS"
    value:{string_value:"no"}

}