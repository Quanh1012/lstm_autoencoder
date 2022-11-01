#!usr/bin/bash
parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

CONFIG_DIR="configs"
acc=95.67
acc_val=96.53
for STR in $CONFIG_DIR/done/*.yml; do
	substring=true;
	for SUB in "$@"; do
		if [[ "$STR" != *"$SUB"* ]]; then
			substring=false;
		fi;
	done;
	if [ "$substring" == true ]; then
		echo "Check the result with config $STR";
		eval $(parse_yaml $STR)
        echo $accuracy_test;
        st=`echo "$accuracy_test >= $acc" | bc`
        stval=`echo "$accuracy_val >= $acc_val" | bc`
        echo $st;
        echo $stval;
        if [[ $st -eq 1  && $stval -eq 1 ]]; then  
         echo "Better accuracy in $STR";
         mv -v $STR $CONFIG_DIR/best_results;
      #   else
      #       rm $STR ;        fi;
        printf "\n";
        printf '=%.0s' {1..100};
        printf "\n";
	fi;
done