mod mnist_model;
use mnist_model::{MnistInput, MnistModel};

use std::sync::{Arc, Mutex};

use actix_rt;
use actix_web::{middleware, web, App, Error, HttpResponse, HttpServer};
use base64;
use env_logger;
use serde::Deserialize;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {
    /// Directory where TensorFlow model is saved
    #[structopt(short, long)]
    model_dir: String,
}

#[derive(Debug, Deserialize)]
struct MnistRequest {
    // Base64 encoded image PNG/JPG
    image: String,
}

async fn predict_mnist(
    model: web::Data<Arc<Mutex<MnistModel>>>,
    data: web::Json<MnistRequest>,
) -> Result<HttpResponse, Error> {
    let image_bytes =
        base64::decode(&data.image).map_err(|e| HttpResponse::BadRequest().body(e.to_string()))?;
    let input = MnistInput::from_image_bytes(image_bytes)
        .map_err(|e| HttpResponse::InternalServerError().body(e.to_string()))?;
    let res = model
        .lock()
        .unwrap()
        .predict(input)
        .map_err(|_| HttpResponse::InternalServerError())?;
    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .json(res))
}

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    let opt = Opt::from_args();
    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    println!("Loading saved model from {}", &opt.model_dir);
    let model = Arc::new(Mutex::new({
        MnistModel::from_dir(&opt.model_dir)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
    }));

    let endpoint = "127.0.0.1:8080";
    println!("Running server at {}", endpoint);
    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .data(web::JsonConfig::default().limit(4096))
            .data(model.clone())
            .service(web::resource("/mnist").route(web::post().to(predict_mnist)))
    })
    .bind(endpoint)?
    .run()
    .await
}
