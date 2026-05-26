# Payments & Fraud — Domain Notes

> Personal learning notes built while constructing the fraud detection system. 
> Written in my own words to consolidate understanding. Will be used as context 
> for the LLM educational agent in week 4.

## Block 1 — Anatomy of a card payment

* Actores
- Cardholder -> Persona que paga
- Merchant -> Comercio quien vende 
- Issuer -> Entidad que le dio la tarjeta al cardholder
- PSP/Acquirer -> Entidad que permite al merchant aceptar pagos con tarjetas 
- Network/schemas -> Conectan al issuer con el Acquirer (Visa, Mastercard, Amex)

* flujo de pago con tarjeta
- Authorization -> Se manda una solicitud al Issuer, este decide si aceptar/rechazar/error (Aca no se mueve la plata)
- Capture -> Si se autoriza, se 'reserva' el monto
- Clearing -> Etapa administrativa donde se prepara la info para que luego se liquide correctamente (monto final, fees, issuer, etc)
- Settelment -> Se mueve la plata, se descuentan los fees y se le llega al merchant 

* Acquirer vs processor vs gateway
- Gateway -> captura y envía la solicitud de pago.
- Acquirer -> permite al comercio aceptar tarjetas y cobrar.
- Processor ->  procesa/enruta técnicamente las transacciones.

## Block 2 — Who makes money in the chain

- Interchange fee -> comision pagada por el merchant al issuer 
- Merchant Discount Rate (MDR) -> El porcentaje total que se le descuenta al merchant por fees antes de impuestos. Vende por 10, le llegan 9, MDR=10%
- Scheme fees -> comision que cobran las redes/scheme al merchant

* Cuentas/wallets y rails de las tarjetas
- Cuentas/wallets -> cuenta digital donde el usuario puede tener saldo, tarjetas, CVU, alias, QR, etc
- Rails -> Vias por donde se mueve la plata. Tarjetas -> Block 1, Saldo Mp -> se mueve dentro de ecosistema/cuenta de MP

- En fraude, el objetivo no es rechazar todo, sino equilibrar fraude evitado vs. ventas legítimas rechazadas.

## Block 3 — Transaction types

- Card Present (CP) -> Pagas en el lugar fisico, pagar con apple pay en un POS se considera CP
- Card Not Present (CNP) -> Cuando no la tarjeta fisicamente, por ej cargas tu tarjeta en una pagina para una compra online

* Capas de autenticacion
- 3DS -> se pide una confirmacion manual al usuario. Ej, Confirmar desde la app de MP. Barrera importante para fraude, mas friccion para el usuario
- 3DS2 -> Busca autenticar con menos friccion, usando contexto como IP, device, historial, etc. 
- Strong Customer Autentication (SCA) -> La idea es validar al usuario usando al menos dos factores de categorías distintas. Ej, Contraseña + Huella

* CIT vs MIT
- Customer Initiated Transaction (CIT) -> Pagas un uber en el momento, Entras a MP y compras algo
- Merchant Iniciated Transaction (MIT) -> Cobro automatico de Netflix, Pago de una cuota programada
- En CIT podes pedir 3DS. En MIT no podes pedirle al usuario todos los meses

## Block 4 — Chargebacks and disputes

* Chargebacks y Disputs
- Chargeback -> una revision forzada de una transaccion con tarjeta. El cliente va al issuer, y este inicia una disputa con el merchant si corresponde. Ej no hice esta compra o no recibi el producto
- Refund -> El merchant voluntariamente devuelve el dinero.

* Flujo tipico
1- El cardholder ve un consumo problemático.
2- Reclama al issuer.
3- El issuer evalúa si hay base para disputa.
4- Si corresponde, inicia el chargeback contra el acquirer.
5- El acquirer se lo pasa al merchant.
6- El merchant decide si acepta la pérdida o pelea el caso.

* Porque CNP es mas delicado para el Merchant
- Como en una compra online ni el cardholder ni la tarjeta estan fisicamente, si hay una disputa, es mas dificil probar que la copra fue legitima
- Por eso 3DS es importante, mueve la responsabilidad del merchant al issuer

## Block 5 — Fraud taxonomy

* Tipos de fraude
- Card testing -> un fraudster prueba distintas tarjetas en distintos comercios con montos bajos para ver cuales funcionan. 
    - BIN attacks -> BIN primeros numeros de la tarjeta. Prueban los siguientes random hasta que alguna funcione
- Account takeover -> Alguien te roba la cuenta. Se puede detectar por el IP, login desde tel nuevo, otro pais, etc
- Synthetic identity fraud -> Alguien crea una identidad con datos falsos y algunos verdaderos. 
    - Ejemplo -> Un fraudster crea varias cuentas en una fintech. Hace movimientos chicos para parecer normal. Después pide crédito y desaparece.
- First-party/friendly fraud -> Un cliente legitimo hace una operacion y actua como si no la hubiese hecho
- Stolen card fraud -> Alguien usa datos de una tarjeta ajena sin autorización. Puede ser robada, datos filtrados, etc

## Block 6 — KYC / AML / Compliance


## Findings (data-grounded observations)

* Velocity
- Velocity feature -> Mide la tasa de actividad de una entidad en un perdiodo de tiempo (transacciones, monto, merchants distintos, etc)
- Por que importa -> El comportamiento legitimo tiene un ritmo. Un account takeover se ve como un cambio brusco de cadencia: una tarjeta que venía a 2 transacciones por semana de repente hace 8 en una tarde
--> Velocity suele estar entre las top 10 mas predictivas en un sistema de fraude

* SHAP -> Metodo para explicar predicciones individuales, le asigna un valor numerico a cada variable dependiendo de cuanto contribuyo a la prediccion
- Ej, una transaccion flageada 0.92, SHAP te dice card1 +0.4, card2 +0.10, ...., eso deberia sumar 0.92
- Porque usarla -> El feature_importance de LightGBM te da resultado globales (ej, cuantas veces se uso el feature en el split). En Fraude se quiere ver xq cada transaccion fue flageada para luego poder ver si rechazar o no.



## Open questions
