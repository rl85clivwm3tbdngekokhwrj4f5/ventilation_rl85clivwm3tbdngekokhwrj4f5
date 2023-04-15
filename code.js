function toggle_method(obj)
{
   let decay_content = document.getElementById('decay_content');
   let steady_content = document.getElementById('steady_content');
   let rise_content = document.getElementById('rise_content');
   decay_content.classList.remove('page_content_active');
   steady_content.classList.remove('page_content_active');
   rise_content.classList.remove('page_content_active');

   if(obj.id === "decay")
   {
      decay_content.classList.add('page_content_active');
   }
   else if(obj.id === "steady")
   {
      steady_content.classList.add('page_content_active');
   }
   else if(obj.id === "build_up")
   {
      rise_content.classList.add('page_content_active');
   }
   else
   {
      alert("Unknown object id: " + obj.id);
   }
}

function add_input(parent_obj_name, min_data_point, input_text_box_name, button_name, id_prefix)
{
   const count = document.getElementById(input_text_box_name).value;
   let parent_obj = document.getElementById(parent_obj_name);
   let button = document.getElementById(button_name);

   remove_input(parent_obj, id_prefix);

   if(count < parseInt(min_data_point))
   {
      alert("Input at least " + min_data_point + " data point");
      return;
   }
   for(let i = 0; i < count; i++)
   {
      let label_time = document.createElement("label");
      let input_time_id = id_prefix + "_input_time_" + i;
      label_time.setAttribute("for", input_time_id);
      label_time.id = id_prefix + "_label_time_" + i;
      let label_time_text = document.createTextNode("Time (minute): ");
      label_time.appendChild(label_time_text);

      let input_time = document.createElement("input");
      input_time.setAttribute("type", "text");
      input_time.id = input_time_id;

      let label_co2 = document.createElement("label");
      let input_co2_id = id_prefix + "_input_co2_" + i;
      label_co2.setAttribute("for", input_co2_id);
      label_co2.id = id_prefix + "_label_co2_" + i;
      let label_co2_text = document.createTextNode("  CO2 level (ppm): ");
      label_co2.appendChild(label_co2_text);

      let input_co2 = document.createElement("input");
      input_co2.setAttribute("type", "text");
      input_co2.id = input_co2_id;

      let br = document.createElement("br");
      br.id = id_prefix + "_br_" + i;
   
      parent_obj.insertBefore(label_time, button);
      parent_obj.insertBefore(input_time, button);
      parent_obj.insertBefore(label_co2, button);
      parent_obj.insertBefore(input_co2, button);
      parent_obj.insertBefore(br, button);
   }
}

function remove_input(parent_obj, id_prefix)
{
   const children = parent_obj.childNodes;
   const child_length = children.length;	
   let delete_children = [];

   for(let i = 0; i < child_length; i++)
   {
      let child_id = children[i].id + "";	   
      if(child_id.includes(id_prefix))
      {
	 delete_children.push(children[i]);  
      }
   }

   const delete_child_length = delete_children.length;
   for(let i = 0; i < delete_child_length; i++)
   {
      delete_children[i].remove();
   }
}

function get_input_values(time_array, co2_array, parent_obj_name, input_tag_prefix)
{
   let data_array = document.getElementById(parent_obj_name).getElementsByTagName("input");
   const data_length = data_array.length;

   for(let i = 0; i < data_length; i++)
   {
      let child_id = data_array[i].id + "";
      if(child_id.includes(input_tag_prefix))
      {
         if(child_id.includes("input_time"))
	 {
            time_array.push(parseFloat(data_array[i].value));		 
	 }
	 else if(child_id.includes("input_co2"))
	 {
            co2_array.push(parseFloat(data_array[i].value));
	 }
	 else
	 {
            alert("Unknown child id: " + child_id);
	 }
      }
   }
   if(time_array.length != co2_array.length)
   {
      alert("Time array and CO2 array length mismatch!");
   }
}

function steady_co2_solve()
{
   let external_co2 = parseFloat(document.getElementById("steady_co2_external").value);
   let steady_co2 = parseFloat(document.getElementById("steady_co2_steady").value);
   let co2_gen = parseFloat(document.getElementById("steady_co2_gen").value);
   let ven_unit;

   if(document.getElementById("steady_co2_unit_cfm").checked)
   {
      ven_unit = "cfm";
   }
   else if(document.getElementById("steady_co2_unit_Ls").checked)
   {
      ven_unit = "L/s";
   }
   else
   {
      alert("Unit not selected!");
   }

   let ventilation = 1.0e6 * co2_gen / (steady_co2 - external_co2);

   let previous_vent = document.getElementById("steady_co2_dynamic_vent");
   if(previous_vent != null)
   {
      previous_vent.remove();
   }
	
   let para_vent = document.createElement("p");
   para_vent.id = "steady_co2_dynamic_vent";
   let vent_text = document.createTextNode("Ventilation flow rate is: " + ventilation + "  " + ven_unit);
   para_vent.appendChild(vent_text);

   let parent_obj = document.getElementById("steady_content");
   parent_obj.appendChild(para_vent);
}

function decay_co2_solve()
{
   let external_co2 = parseFloat(document.getElementById("decay_co2_external").value);
   let time_array = [];
   let co2_array = [];
   let sum_time = 0.0;
   let sum_time_squared = 0.0;
   let sum_co2 = 0.0;
   let sum_co2_time = 0.0; 

   get_input_values(time_array, co2_array, "decay_content", "decay_co2_dynamic");

   let time_transform_array = time_array.slice();
   let co2_transform_array = co2_array.slice();
   if(time_transform_array.length != co2_transform_array.length)
   {
      alert("Array length mismatch!");
   }
   const count = time_transform_array.length;

   for(let i = 0; i < count; i++)
   {
      time_transform_array[i] = time_transform_array[i] / 60.0;
   }

   for(let i = 0; i < count; i++)
   {
      co2_transform_array[i] = Math.log(co2_transform_array[i] - external_co2);
   }

   for(let i = 0; i < count; i++)
   {
      sum_time += time_transform_array[i];
      sum_time_squared += (time_transform_array[i] * time_transform_array[i]);
   }

   for(let i = 0; i < count; i++)
   {
      sum_co2 += co2_transform_array[i];
      sum_co2_time += (co2_transform_array[i] * time_transform_array[i]);
   }

   let slope = (count * sum_co2_time - sum_time * sum_co2) / (count * sum_time_squared - sum_time * sum_time);
   let ACH = -slope;

   let corr_r_squared = compute_correlation(time_transform_array, co2_transform_array);

   let previous_ACH = document.getElementById("decay_co2_dynamic_ACH");
   if(previous_ACH != null)
   {
      previous_ACH.remove();
   }
   let previous_corr = document.getElementById("decay_co2_dynamic_corr");
   if(previous_corr != null)
   {
      previous_corr.remove();
   }

   let para_ACH = document.createElement("p");
   para_ACH.id = "decay_co2_dynamic_ACH";
   let ACH_text = document.createTextNode("ACH is: " + ACH);
   para_ACH.appendChild(ACH_text);

   let para_corr = document.createElement("p");
   para_corr.id = "decay_co2_dynamic_corr";
   let corr_text1 = document.createTextNode("Correlation r");
   let corr_text2 = document.createTextNode("2");
   let corr_text3 = document.createTextNode(" is: " + corr_r_squared);
   let sup_corr = document.createElement("sup");
   sup_corr.appendChild(corr_text2);
   para_corr.appendChild(corr_text1);
   para_corr.appendChild(sup_corr);
   para_corr.appendChild(corr_text3);

   let parent_obj = document.getElementById("decay_content");	
   parent_obj.appendChild(para_ACH);
   parent_obj.appendChild(para_corr);
}

function rise_co2_solve()
{
   let time_array = [];
   let co2_array = [];
   let sum_time = 0.0;
   let sum_time_squared = 0.0;

   get_input_values(time_array, co2_array, "rise_content", "rise_co2_dynamic");

   let time_transform_array = time_array.slice();
   let co2_transform_array = co2_array.slice();
   if(time_transform_array.length != co2_transform_array.length)
   {
      alert("Array length mismatch!");
   }
   const count = time_transform_array.length;

   for(let i = 0; i < count; i++)
   {
      time_transform_array[i] = (time_transform_array[i] - time_array[0]) / 60.0;
   }

   for(let i = 0; i < count; i++)
   {
      sum_time += time_transform_array[i];
      sum_time_squared += (time_transform_array[i] * time_transform_array[i]);
   }

   let CO2_vector_function = 
      function(input_vector, output_vector)
      {
         const size = input_vector.length;
         if(size != 2 || output_vector.length != size)
         {
            alert("Unexpected size!");
         }
         let sum_co2 = 0.0;
	 let sum_co2_recip = 0.0;
	 let sum_time_co2_recip = 0.0;
	 let sum_co2_time = 0.0;
	 let ln_co2_0 = Math.log(input_vector[0] - co2_array[0]);
	 let co2_recip_0 = 1.0 / (input_vector[0] - co2_array[0]);
	 for(let i = 0; i < count; i++)
	 {
            let tmp = Math.log(input_vector[0] - co2_array[i]);
            let tmp2 = 1.0 / (input_vector[0] - co2_array[i]);
            sum_co2 += (tmp * (tmp2 - co2_recip_0));
            sum_co2_recip += tmp2;
            sum_time_co2_recip += (time_transform_array[i] * tmp2);
            sum_co2_time += (time_transform_array[i] * tmp);
	 }
	 output_vector[0] = sum_co2 - ln_co2_0 * (sum_co2_recip - count * co2_recip_0) + input_vector[1] * (sum_time_co2_recip - co2_recip_0 * sum_time);
         output_vector[1] = sum_co2_time - ln_co2_0 * sum_time + input_vector[1] * sum_time_squared;
      };
   let CO2_Jacobian_matrix =
      function(input_vector, output_matrix)
      {
         const size = input_vector.length;
         if(size != 2 || output_matrix.length != size || output_matrix[0].length != size || output_matrix[1].length != size)
	 {
            alert("Unexpected size!");
	 }
         let sum_co2_cross = 0.0;
         let sum_co2_co2_recip_squ = 0.0;
         let sum_co2_recip = 0.0;
         let sum_co2_recip_squ = 0.0;
         let sum_time_co2_recip_squ = 0.0;
         let sum_time_co2_recip = 0.0;
	 let ln_co2_0 = Math.log(input_vector[0] - co2_array[0]);
         let co2_recip_0 = 1.0 / (input_vector[0] - co2_array[0]);
	 for(let i = 0; i < count; i++)
         {
            let tmp = Math.log(input_vector[0] - co2_array[i]);
            let tmp2 = 1.0 / (input_vector[0] - co2_array[i]);
            let tmp3 = co2_recip_0 * co2_recip_0 - tmp2 * tmp2;
            sum_co2_cross += (tmp2 * tmp2 - tmp2 * co2_recip_0);
            sum_co2_co2_recip_squ += (tmp * tmp3);
            sum_co2_recip += (tmp2 - co2_recip_0);
            sum_co2_recip_squ += tmp3;
            sum_time_co2_recip_squ += (time_transform_array[i] * tmp3);
            sum_time_co2_recip += (time_transform_array[i] * tmp2);
	 }
	 output_matrix[0][0] = sum_co2_cross + sum_co2_co2_recip_squ - co2_recip_0 * sum_co2_recip - ln_co2_0 * sum_co2_recip_squ + input_vector[1] * sum_time_co2_recip_squ;
	 output_matrix[0][1] = sum_time_co2_recip - co2_recip_0 * sum_time;

         output_matrix[1][0] = sum_time_co2_recip - co2_recip_0 * sum_time;
         output_matrix[1][1] = sum_time_squared;
      };
   let CO2_residual_function = 
      function(input_vector)
      {
         if(input_vector.length != 2)
         {
            alert("Unexpected size!");
         }

         let ln_co2_0 = Math.log(input_vector[0] - co2_array[0]);
         let sum = 0.0;
         for(let i = 0; i < count; i++)
         {
            let tmp = Math.log(input_vector[0] - co2_array[i]);
            let diff = (tmp - (ln_co2_0 - input_vector[1] * time_transform_array[i]));
            sum += (diff * diff);
	 }
         return sum;
      };

   let max_val = 0.0;
   for(let i = 0; i < count; i++)
   {
      if(co2_array[i] > max_val)
      {
         max_val = co2_array[i];
      }
   }
   let vector_X = [max_val + 5.0, 1.0]; // initial guess
   let vector_X_out = [0.0, 0.0];

   if(gradient_descent(CO2_vector_function, CO2_Jacobian_matrix, vector_X, vector_X_out, 1.0e-3, 250) == false)
   {
      alert("gradient descent maximum iterations exceeded.");
   }
   vector_X[0] = vector_X_out[0];
   vector_X[1] = vector_X_out[1];

   if(Newton_Raphson(CO2_vector_function, CO2_Jacobian_matrix, vector_X, vector_X_out, 1.0e-12, 120) == false)
   {
      alert("Newton-Raphson did not converge to specified tolerance.");
   }

   let CO2_steady = vector_X_out[0];
   let ACH = vector_X_out[1];
   let residual = CO2_residual_function(vector_X_out);

   for(let i = 0; i < count; i++)
   {
      co2_transform_array[i] = Math.log(CO2_steady - co2_array[i]);
   }

   let corr_r_squared = compute_correlation(time_transform_array, co2_transform_array);

   let previous_CO2_steady = document.getElementById("rise_co2_dynamic_steady");
   if(previous_CO2_steady != null)
   {
      previous_CO2_steady.remove();
   }
   let previous_ACH = document.getElementById("rise_co2_dynamic_ACH");
   if(previous_ACH != null)
   {
      previous_ACH.remove();
   }
   let previous_residual = document.getElementById("rise_co2_dynamic_residual");
   if(previous_residual != null)
   {
      previous_residual.remove();
   }
   let previous_corr = document.getElementById("rise_co2_dynamic_corr");
   if(previous_corr != null)
   {
      previous_corr.remove();
   }

   let para_CO2_steady = document.createElement("p");
   para_CO2_steady.id = "rise_co2_dynamic_steady";
   let CO2_steady_text = document.createTextNode("Computed steady state CO2 is: " + CO2_steady + " ppm");
   para_CO2_steady.appendChild(CO2_steady_text);

   let para_ACH = document.createElement("p");
   para_ACH.id = "rise_co2_dynamic_ACH";
   let ACH_text = document.createTextNode("Computed ACH is: " + ACH);
   para_ACH.appendChild(ACH_text);

   let para_residual = document.createElement("p");
   para_residual.id = "rise_co2_dynamic_residual";
   let residual_text = document.createTextNode("Sum of squared residual is: " + residual);
   para_residual.appendChild(residual_text);

   let para_corr = document.createElement("p");
   para_corr.id = "rise_co2_dynamic_corr";
   let corr_text1 = document.createTextNode("Correlation r");
   let corr_text2 = document.createTextNode("2");
   let corr_text3 = document.createTextNode(" is: " + corr_r_squared);
   let sup_corr = document.createElement("sup");
   sup_corr.appendChild(corr_text2);
   para_corr.appendChild(corr_text1);
   para_corr.appendChild(sup_corr);
   para_corr.appendChild(corr_text3);

   let parent_obj = document.getElementById("rise_content");
   parent_obj.appendChild(para_CO2_steady);
   parent_obj.appendChild(para_ACH);
   parent_obj.appendChild(para_residual);
   parent_obj.appendChild(para_corr);
}

function compute_correlation(x_array, y_array)
{
   if(x_array.length != y_array.length)
   {
      alert("Array length mismatch!");
   }
   const count = x_array.length;

   let sum_x = 0.0;
   let sum_y = 0.0;

   for(let i = 0; i < count; i++)
   {
      sum_x += x_array[i];
      sum_y += y_array[i];
   }
   let mean_x = sum_x / count;
   let mean_y = sum_y / count;

   let sum_x_mean_squ = 0.0;
   let sum_y_mean_squ = 0.0;
   let sum_x_mean_y_mean = 0.0;

   for(let i = 0; i < count; i++)
   {
      sum_x_mean_squ += ((x_array[i] - mean_x) * (x_array[i] - mean_x));
      sum_y_mean_squ += ((y_array[i] - mean_y) * (y_array[i] - mean_y));
      sum_x_mean_y_mean += ((x_array[i] - mean_x) * (y_array[i] - mean_y));
   }
   let corr_r = sum_x_mean_y_mean / (Math.sqrt(sum_x_mean_squ) * Math.sqrt(sum_y_mean_squ));
   let corr_r_squ = corr_r * corr_r;
   return corr_r_squ;	
}

// solve the linear system Ax=B for x, given matrix A and vector B
function Gauss_Jordan(LHS_matrix_A, RHS_vector_B, result_vector_x)
{
   const size = RHS_vector_B.length;

   if(size === 0 || result_vector_x.length != size || LHS_matrix_A.length != size)
   {
      alert("Unexpected size!");
   }
   for(let i = 0; i < size; i++)
   {
      if(LHS_matrix_A[i].length != size)
      {
         alert("Unexpected size!");
      }
   }

   let matrix_A = [...Array(size)].map( () => Array(size).fill(0.0));
   let vector_B = Array(size).fill(0.0);

   for(let i = 0; i < size; i++)
   {
      for(let j = 0; j < size; j++)
      {
         matrix_A[i][j] = LHS_matrix_A[i][j];
      }
   }
   for(let i = 0; i < size; i++)
   {
      vector_B[i] = RHS_vector_B[i];
   }

   for(let i = 0; i < size; i++)
   {
      let pivot_row = i;
      for(let j = i+1; j < size; j++)
      {
	 if(Math.abs(matrix_A[j][i]) > Math.abs(matrix_A[pivot_row][i]))
	 {
            pivot_row = j;
	 }
      }
      if(pivot_row != i)
      {
         //partial pivoting
	 let tmp;
	 for(let j = 0; j < size; j++)
	 {
	    tmp = matrix_A[i][j];
            matrix_A[i][j] = matrix_A[pivot_row][j];
	    matrix_A[pivot_row][j] = tmp;
	 }
	 tmp = vector_B[i];
	 vector_B[i] = vector_B[pivot_row];
	 vector_B[pivot_row] = tmp;
      }

      for(let j = i + 1; j < size; j++)
      {
         let multiplier = -(matrix_A[j][i] / matrix_A[i][i]);
	 for(let k = i; k < size; k++)
         {
            matrix_A[j][k] += (matrix_A[i][k] * multiplier);
         }
	 vector_B[j] += (vector_B[i] * multiplier);
      }
   }
   
   for(let i = size - 1; i >= 0; i--)
   {
      for(let j = i - 1; j >= 0; j--)
      {
         let multiplier = -(matrix_A[j][i] / matrix_A[i][i]);
	 matrix_A[j][i] += (matrix_A[i][i] * multiplier);
	 vector_B[j] += (vector_B[i] * multiplier);
      }
   }

   for(let i = 0; i < size; i++)
   {
      multiplier = 1.0 / matrix_A[i][i];
      matrix_A[i][i] *= multiplier;
      vector_B[i] *= multiplier;
   }

   for(let i = 0; i < size; i++)
   {
      result_vector_x[i] = vector_B[i];
   }
}

// assume function_F is a function parameter with prototype: function_F(input_array, output_array)
// assume jacobian_J is a function parameter with prototype: jacobian_J(input_array, output_matrix)
function Newton_Raphson(function_F, jacobian_J, initial_guess_vector, result_vector, tolerance, max_iterations)
{
   let iter_count = 0;
   const size = initial_guess_vector.length;
   if(result_vector.length != size)
   {
      alert("Unexpected size!");
   }

   let vector_X = Array(size).fill(0.0);
   let vector_Fx = Array(size).fill(0.0);
   let matrix_Jx = [...Array(size)].map( () => Array(size).fill(0.0));
   let vector_D = Array(size).fill(0.0);

   for(let i = 0; i < size; i++)
   {
      vector_X[i] = initial_guess_vector[i];
   }

   for(iter_count = 0; iter_count < max_iterations; iter_count++)
   {
      function_F(vector_X, vector_Fx);
      jacobian_J(vector_X, matrix_Jx);
      for(let i = 0; i < size; i++)
      {
         vector_Fx[i] = -vector_Fx[i];
      }
      Gauss_Jordan(matrix_Jx, vector_Fx, vector_D);
      for(let i = 0; i < size; i++)
      {
         vector_X[i] += vector_D[i];
      }
      let sum_D = 0.0;
      let sum_X = 0.0;
      // use L1 norm
      for(let i = 0; i < size; i++)
      {
         sum_D += Math.abs(vector_D[i]);
         sum_X += Math.abs(vector_X[i]);
      }
      if( (sum_D / sum_X) < tolerance)
      {
	 for(let i = 0; i < size; i++)
	 {
            result_vector[i] = vector_X[i];
	 }
         return true;
      }
   }
   for(let i = 0; i < size; i++)
   {
      result_vector[i] = vector_X[i];
   }
   return false;
}

// assume function_F is a function parameter with prototype: function_F(input_array, output_array)
// assume jacobian_J is a function parameter with prototype: jacobian_J(input_array, output_matrix)
function gradient_descent(function_F, jacobian_J, initial_guess_vector, result_vector, tolerance, max_iterations)
{
   let iter_count = 0;
   const size = initial_guess_vector.length;
   if(result_vector.length != size)
   {
      alert("Unexpected size!");
   }

   let vector_X = Array(size).fill(0.0);
   let vector_X_tmp = Array(size).fill(0.0);
   let vector_Fx = Array(size).fill(0.0);
   let matrix_Jx = [...Array(size)].map( () => Array(size).fill(0.0));
   let aux_func_g_val = 0.0;
   let aux_func_g_val_tmp = 0.0;
   let gradient_vector = Array(size).fill(0.0);
   let gradient_multiplier = 0.0;
   let gradient_multiplier_tmp = 0.0;
   let aux_func_g =
      function(vector_input)
      {
         let sum = 0.0;
         for(let i = 0; i < size; i++)
         {
            sum += (vector_input[i] * vector_input[i]);
	 }
	 return (0.5 * sum);
      };
   let aux_func_vector_X =
      function(input_vector, gradient_vector_input, grad_multiplier, output_vector)
      {
         for(let i = 0; i < size; i++)
         {
            output_vector[i] = input_vector[i] - grad_multiplier * gradient_vector_input[i];
	 }
      };

   for(let i = 0; i < size; i++)
   {
      vector_X[i] = initial_guess_vector[i];
   }

   for(iter_count = 0; iter_count < max_iterations; iter_count++)
   {
      function_F(vector_X, vector_Fx);
      aux_func_g_val = aux_func_g(vector_Fx);
      jacobian_J(vector_X, matrix_Jx);
      for(let i = 0; i < size; i++)
      {
         let sum = 0.0;
	 for(let j = 0; j < size; j++)
	 {
            sum += (matrix_Jx[j][i] * vector_Fx[j]);
	 }
	 gradient_vector[i] = sum;
      }
      //L2 norm
      let norm = 0.0;
      for(let i = 0; i < size; i++)
      {
         norm += (gradient_vector[i] * gradient_vector[i]);
      }
      norm = Math.sqrt(norm);
      for(let i = 0; i < size; i++)
      {
         gradient_vector[i] = gradient_vector[i] / norm;
      }
      gradient_multiplier = 0.1;
      aux_func_vector_X(vector_X, gradient_vector, gradient_multiplier, vector_X_tmp);
      function_F(vector_X_tmp, vector_Fx);
      aux_func_g_val_tmp = aux_func_g(vector_Fx);
      if(isNaN(aux_func_g_val_tmp) || aux_func_g_val_tmp > aux_func_g_val)
      {
         while(isNaN(aux_func_g_val_tmp) || aux_func_g_val_tmp > aux_func_g_val)
         {
            gradient_multiplier *= 0.5;
	    aux_func_vector_X(vector_X, gradient_vector, gradient_multiplier, vector_X_tmp);
            function_F(vector_X_tmp, vector_Fx);
            aux_func_g_val_tmp = aux_func_g(vector_Fx);
	    if(gradient_multiplier < tolerance && aux_func_g_val_tmp < tolerance)
	    {
               for(let i = 0; i < size; i++)
               {
                  result_vector[i] = vector_X[i];
	       }
               return true;
	    }
         }
      }
      else
      {
         while(!isNaN(aux_func_g_val_tmp) && aux_func_g_val_tmp < aux_func_g_val)
         {
            aux_func_g_val = aux_func_g_val_tmp;
            gradient_multiplier_tmp = gradient_multiplier * 2.0;
            aux_func_vector_X(vector_X, gradient_vector, gradient_multiplier_tmp, vector_X_tmp);
            function_F(vector_X_tmp, vector_Fx);
            aux_func_g_val_tmp = aux_func_g(vector_Fx);
            if(!isNaN(aux_func_g_val_tmp) && aux_func_g_val_tmp < aux_func_g_val)
            {
               gradient_multiplier = gradient_multiplier_tmp;
	    }
	 }
      }
      aux_func_vector_X(vector_X, gradient_vector, gradient_multiplier, vector_X);
      if(gradient_multiplier < tolerance && aux_func_g_val < tolerance)
      {
         for(let i = 0; i < size; i++)
         {
            result_vector[i] = vector_X[i];
         }
         return true;
      }
   }
   for(let i = 0; i < size; i++)
   {
      result_vector[i] = vector_X[i];
   }
   return false;
}
